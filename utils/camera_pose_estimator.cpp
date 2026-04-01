#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vtkObject.h>

struct Pose
{
    double timestamp; // combined seconds + nanoseconds
    double px, py, pz;
    double qx, qy, qz, qw;
};

struct ImageEntry
{
    int index;
    std::string filename;
    double timestamp;
};

double combine_timestamp(int64_t sec, uint32_t nsec)
{
    return static_cast<double>(sec) + static_cast<double>(nsec) * 1e-9;
}

// Quaternion SLERP
void quat_slerp(double t,
                 double ax, double ay, double az, double aw,
                 double bx, double by, double bz, double bw,
                 double &rx, double &ry, double &rz, double &rw)
{
    // Compute dot product
    double dot = ax * bx + ay * by + az * bz + aw * bw;

    // If dot is negative, negate one quaternion to take the shorter path
    if (dot < 0.0)
    {
        bx = -bx; by = -by; bz = -bz; bw = -bw;
        dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995)
    {
        rx = ax + t * (bx - ax);
        ry = ay + t * (by - ay);
        rz = az + t * (bz - az);
        rw = aw + t * (bw - aw);
    }
    else
    {
        double theta = std::acos(dot);
        double sin_theta = std::sin(theta);
        double wa = std::sin((1.0 - t) * theta) / sin_theta;
        double wb = std::sin(t * theta) / sin_theta;

        rx = wa * ax + wb * bx;
        ry = wa * ay + wb * by;
        rz = wa * az + wb * bz;
        rw = wa * aw + wb * bw;
    }

    // Normalize
    double norm = std::sqrt(rx * rx + ry * ry + rz * rz + rw * rw);
    rx /= norm; ry /= norm; rz /= norm; rw /= norm;
}

std::vector<Pose> read_odometry_csv(const std::string & path)
{
    std::vector<Pose> poses;
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open odometry file '" << path << "'" << std::endl;
        return poses;
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        Pose p;

        int64_t sec;
        uint32_t nsec;

        std::getline(ss, token, ','); // index
        std::getline(ss, token, ','); sec = std::stoll(token);
        std::getline(ss, token, ','); nsec = std::stoul(token);
        std::getline(ss, token, ','); p.px = std::stod(token);
        std::getline(ss, token, ','); p.py = std::stod(token);
        std::getline(ss, token, ','); p.pz = std::stod(token);
        std::getline(ss, token, ','); p.qx = std::stod(token);
        std::getline(ss, token, ','); p.qy = std::stod(token);
        std::getline(ss, token, ','); p.qz = std::stod(token);
        std::getline(ss, token, ','); p.qw = std::stod(token);

        p.timestamp = combine_timestamp(sec, nsec);
        poses.push_back(p);
    }

    return poses;
}

std::vector<ImageEntry> read_image_csv(const std::string & path)
{
    std::vector<ImageEntry> entries;
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open image timestamp file '" << path << "'" << std::endl;
        return entries;
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        ImageEntry e;

        int64_t sec;
        uint32_t nsec;

        std::getline(ss, token, ','); e.index = std::stoi(token);
        std::getline(ss, token, ','); e.filename = token;
        std::getline(ss, token, ','); sec = std::stoll(token);
        std::getline(ss, token, ','); nsec = std::stoul(token);

        e.timestamp = combine_timestamp(sec, nsec);
        entries.push_back(e);
    }

    return entries;
}

// Map a normalized value [0,1] to an RGB color using a turbo-like colormap
void rainbow_colormap(float t, uint8_t &r, uint8_t &g, uint8_t &b)
{
    t = std::clamp(t, 0.0f, 1.0f);

    // HSV with hue from 270 (blue) down to 0 (red) as height increases
    float hue = (1.0f - t) * 270.0f;
    float s = 1.0f, v = 1.0f;

    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float rf, gf, bf;
    if      (hue < 60)  { rf = c; gf = x; bf = 0; }
    else if (hue < 120) { rf = x; gf = c; bf = 0; }
    else if (hue < 180) { rf = 0; gf = c; bf = x; }
    else if (hue < 240) { rf = 0; gf = x; bf = c; }
    else                { rf = x; gf = 0; bf = c; }

    r = static_cast<uint8_t>((rf + m) * 255.0f);
    g = static_cast<uint8_t>((gf + m) * 255.0f);
    b = static_cast<uint8_t>((bf + m) * 255.0f);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <data_folder> [--visualize]" << std::endl;
        return 1;
    }

    std::string data_folder = argv[1];
    std::string image_csv_path = data_folder + "timestamps/image_timestamps.csv";
    std::string odom_csv_path = data_folder + "timestamps/odom_timestamps.csv";
    std::string output_path = data_folder + "poses/image_poses.csv";
    std::string point_cloud_path = data_folder + "pcd/global_point_cloud.pcd";

    bool visualize = (argc >= 3 && std::string(argv[2]) == "--visualize");

    //create output directory if it doesn't exist
    std::string output_dir = data_folder + "poses";
    std::system(("mkdir -p " + output_dir).c_str());

    // Read input data
    auto images = read_image_csv(image_csv_path);
    auto odom = read_odometry_csv(odom_csv_path);

    if (images.empty())
    {
        std::cerr << "No image entries loaded." << std::endl;
        return 1;
    }
    if (odom.size() < 2)
    {
        std::cerr << "Need at least 2 odometry entries for interpolation." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << images.size() << " image timestamps" << std::endl;
    std::cout << "Loaded " << odom.size() << " odometry poses" << std::endl;
    std::cout << "Odom time range: [" << std::fixed << std::setprecision(6)
              << odom.front().timestamp << ", " << odom.back().timestamp << "]" << std::endl;

    // Ensure odometry is sorted by timestamp
    std::sort(odom.begin(), odom.end(),
        [](const Pose & a, const Pose & b) { return a.timestamp < b.timestamp; });

    // Interpolate pose for each image
    std::ofstream out(output_path);
    if (!out.is_open())
    {
        std::cerr << "Error: Could not open output file '" << output_path << "'" << std::endl;
        return 1;
    }

    out << "index,filename,timestamp,px,py,pz,qx,qy,qz,qw\n";
    out << std::fixed << std::setprecision(12);

    int matched = 0;
    int skipped = 0;
    std::vector<Pose> interpolated_poses;

    for (const auto & img : images)
    {
        // Binary search for the first odom entry with timestamp >= image timestamp
        auto it = std::lower_bound(odom.begin(), odom.end(), img.timestamp,
            [](const Pose & p, double t) { return p.timestamp < t; });

        // Handle edge cases
        if (it == odom.begin())
        {
            // Image is before first odom — use first pose (no extrapolation)
            std::cerr << "Warning: Image '" << img.filename
                      << "' is before odometry range, using first pose." << std::endl;
            const auto & p = odom.front();
            out << img.index << "," << img.filename << "," << img.timestamp << ","
                << p.px << "," << p.py << "," << p.pz << ","
                << p.qx << "," << p.qy << "," << p.qz << "," << p.qw << "\n";
            skipped++;
            continue;
        }
        if (it == odom.end())
        {
            // Image is after last odom — use last pose
            std::cerr << "Warning: Image '" << img.filename
                      << "' is after odometry range, using last pose." << std::endl;
            const auto & p = odom.back();
            out << img.index << "," << img.filename << "," << img.timestamp << ","
                << p.px << "," << p.py << "," << p.pz << ","
                << p.qx << "," << p.qy << "," << p.qz << "," << p.qw << "\n";
            skipped++;
            continue;
        }

        // We have two bounding odom entries
        const Pose & after = *it;
        const Pose & before = *std::prev(it);

        // Compute interpolation factor
        double dt = after.timestamp - before.timestamp;
        double t = (dt > 1e-12) ? (img.timestamp - before.timestamp) / dt : 0.0;

        // Linearly interpolate position
        double px = before.px + t * (after.px - before.px);
        double py = before.py + t * (after.py - before.py);
        double pz = before.pz + t * (after.pz - before.pz);

        // SLERP quaternion orientation
        double qx, qy, qz, qw;
        quat_slerp(t,
                    before.qx, before.qy, before.qz, before.qw,
                    after.qx, after.qy, after.qz, after.qw,
                    qx, qy, qz, qw);

        out << img.index << "," << img.filename << "," << img.timestamp << ","
            << px << "," << py << "," << pz << ","
            << qx << "," << qy << "," << qz << "," << qw << "\n";

        interpolated_poses.push_back({img.timestamp, px, py, pz, qx, qy, qz, qw});
        matched++;
        
    }

    out.close();

    std::cout << "\nResults: " << matched << " interpolated, " << skipped
              << " at boundary (clamped)" << std::endl;
    std::cout << "Saved to '" << output_path << "'" << std::endl;

    // Optional visualization
    if (!visualize)
        return 0;

    vtkObject::GlobalWarningDisplayOff();
    pcl::visualization::PCLVisualizer viewer("Pose Interpolation");
    viewer.setBackgroundColor(0.05, 0.05, 0.05);
    viewer.addCoordinateSystem(1.0);

    // --- Batch odometry path into a single point cloud + line polygon ---
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr odom_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (const auto & p : odom)
    {
        pcl::PointXYZRGB pt;
        pt.x = p.px; pt.y = p.py; pt.z = p.pz;
        pt.r = 80; pt.g = 80; pt.b = 200;
        odom_cloud->push_back(pt);
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> odom_handler(odom_cloud);
    viewer.addPointCloud(odom_cloud, odom_handler, "odom_points");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "odom_points");

    // Odometry path as a single polyline shape
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr path_pts(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto & p : odom)
            path_pts->push_back(pcl::PointXYZ(p.px, p.py, p.pz));

        pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
        pcl::toPCLPointCloud2(*path_pts, mesh->cloud);
        pcl::Vertices strip;
        for (uint32_t i = 0; i < path_pts->size(); i++)
            strip.vertices.push_back(i);
        mesh->polygons.push_back(strip);
        viewer.addPolylineFromPolygonMesh(*mesh, "odom_path");
        viewer.setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_COLOR, 0.3, 0.3, 0.8, "odom_path");
    }

    // --- Batch all frustum lines into a single point cloud with line segments ---
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr frustum_pts(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<pcl::Vertices> lines;

        double frustum_scale = 0.3;
        double aspect = 16.0 / 9.0;

        for (const auto & p : interpolated_poses)
        {
            double qx = p.qx, qy = p.qy, qz = p.qz, qw = p.qw;

            double fx = 1.0 - 2.0 * (qy * qy + qz * qz);
            double fy = 2.0 * (qx * qy + qw * qz);
            double fz = 2.0 * (qx * qz - qw * qy);

            double rx = -2.0 * (qx * qy - qw * qz);
            double ry = -(1.0 - 2.0 * (qx * qx + qz * qz));
            double rz = -2.0 * (qy * qz + qw * qx);

            double ux = 2.0 * (qx * qz + qw * qy);
            double uy = 2.0 * (qy * qz - qw * qx);
            double uz = 1.0 - 2.0 * (qx * qx + qy * qy);

            double hw = frustum_scale * aspect * 0.5;
            double hh = frustum_scale * 0.5;
            double d = frustum_scale;

            uint32_t base = frustum_pts->size();

            // 0: origin
            frustum_pts->push_back(pcl::PointXYZ(p.px, p.py, p.pz));
            // 1: top-left
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d * fx - hw * rx + hh * ux,
                p.py + d * fy - hw * ry + hh * uy,
                p.pz + d * fz - hw * rz + hh * uz));
            // 2: top-right
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d * fx + hw * rx + hh * ux,
                p.py + d * fy + hw * ry + hh * uy,
                p.pz + d * fz + hw * rz + hh * uz));
            // 3: bottom-left
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d * fx - hw * rx - hh * ux,
                p.py + d * fy - hw * ry - hh * uy,
                p.pz + d * fz - hw * rz - hh * uz));
            // 4: bottom-right
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d * fx + hw * rx - hh * ux,
                p.py + d * fy + hw * ry - hh * uy,
                p.pz + d * fz + hw * rz - hh * uz));

            // 8 line segments per frustum: origin to corners + rectangle
            auto add_line = [&](uint32_t a, uint32_t b) {
                pcl::Vertices v;
                v.vertices.push_back(base + a);
                v.vertices.push_back(base + b);
                lines.push_back(v);
            };

            add_line(0, 1); add_line(0, 2); add_line(0, 3); add_line(0, 4);
            add_line(1, 2); add_line(2, 4); add_line(4, 3); add_line(3, 1);
        }
        //load the pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(point_cloud_path, *cloud) == -1)
        {
            std::cerr << "Error: Could not load point cloud file '" << point_cloud_path << "'" << std::endl;
            return 1;
        }

        // Build a colored cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        colored_cloud->resize(cloud->size());
        // Find Z range
        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::lowest();
        for (const auto & pt : cloud->points)
        {
            if (std::isfinite(pt.z))
            {
                z_min = std::min(z_min, pt.z);
                z_max = std::max(z_max, pt.z);
            }
        }

        float z_range = z_max - z_min;
        if (z_range < 1e-6f) z_range = 1.0f;

        std::cout << "Z range: [" << z_min << ", " << z_max << "]" << std::endl;

        for (size_t i = 0; i < cloud->size(); i++)
        {
            const auto & pt = cloud->points[i];
            auto & cpt = colored_cloud->points[i];
            cpt.x = pt.x; cpt.y = pt.y; cpt.z = pt.z;

            float t = (pt.z - z_min) / z_range;
            rainbow_colormap(t, cpt.r, cpt.g, cpt.b);
        }

        // Build a single polygon mesh with all line segments
        pcl::PolygonMesh mesh;
        pcl::toPCLPointCloud2(*frustum_pts, mesh.cloud);
        mesh.polygons = lines;
        viewer.addPolylineFromPolygonMesh(mesh, "frustums");
        viewer.setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.3, 0.3, "frustums");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler(colored_cloud);
        viewer.addPointCloud<pcl::PointXYZRGB>(colored_cloud, rgb_handler, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    }

    std::cout << "\nVisualization:" << std::endl;
    std::cout << "  Blue line + points = odometry path" << std::endl;
    std::cout << "  Red frustums = interpolated image poses + view direction" << std::endl;

    viewer.spin();

    return 0;
}