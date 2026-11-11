#include "CADPreprocessor.h"
#include <open3d/Open3D.h>
#include <QFileInfo>
#include <QDir>
#include <QFile>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

double CADPreprocessor::readSourceUnitScale(const std::string& path) {
    std::vector<std::string> candidates = {
        path + ".unit",
    };
    QFileInfo fi(QString::fromStdString(path));
    QDir dir = fi.dir();
    QString baseName = fi.baseName();
    candidates.push_back(dir.absoluteFilePath(baseName + ".unit").toStdString());
    candidates.push_back(dir.absoluteFilePath(baseName + ".ply.unit").toStdString());
    candidates.push_back(dir.absoluteFilePath(baseName + ".stl.unit").toStdString());
    candidates.push_back(dir.absoluteFilePath(baseName + ".obj.unit").toStdString());

    std::ifstream in;
    std::string foundPath;
    for (const auto& c : candidates) {
        in.open(c);
        if (in.is_open()) {
            foundPath = c;
            break;
        }
    }

    if (!in.is_open()) {
        return -1.0; // No explicit unit file found -- triggers auto-detection based on bounding diagonal
    }

    std::string token;
    in >> token;
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (token == "m" || token == "meter" || token == "meters") return 1.0;
    if (token == "cm" || token == "centimeter" || token == "centimeters") return 0.01;
    if (token == "mm" || token == "millimeter" || token == "millimeters") return 0.001;
    if (token == "in" || token == "inch" || token == "inches") return 0.0254;

    std::cerr << "CADPreprocessor: unrecognized unit '" << token << "' in " << foundPath
              << " -- assuming meters. Expected one of: m, cm, mm, in.\n";
    return 1.0;
}

CADAsset CADPreprocessor::process(const std::string& path, size_t targetPointCount,
                                   double neighborhoodRadius) {
    std::string cachePath = path + ".cadasset";
    double unitScale = readSourceUnitScale(path);
    bool unitExplicitlyFound = (unitScale > 0.0);
    if (!unitExplicitlyFound) unitScale = 1.0; // provisional fallback

    CADAsset asset;
    if (asset.loadFromFile(cachePath)) {
        // Reject stale/unreasonable cache: tabletop parts cannot be > 10m unless explicit
        bool staleGiant = (!unitExplicitlyFound && asset.realBoundingDiag > 10.0);
        if (staleGiant ||
            std::abs(asset.sourceUnitScale - unitScale) > 1e-9 ||
            asset.sampledFeatures.size() != targetPointCount ||
            std::abs(asset.sampleRadius - neighborhoodRadius) > 1e-9) {
            std::cerr << "CADPreprocessor: cache parameters mismatch or stale giant size for "
                      << path << " -- rebuilding.\n";
        } else {
            // Ensure canonical hierarchy is populated
            QFileInfo cadFi(QString::fromStdString(path));
            std::string cadDir = cadFi.dir().absolutePath().toStdString();
            std::string cadName = cadFi.baseName().toStdString();
            std::string triNpz = cadDir + "/" + cadName + "_triangles.npz";

            if (asset.hierarchy.triangles.empty() && asset.normalizedMesh) {
                open3d::geometry::TriangleMesh metricMesh = *asset.normalizedMesh;
                metricMesh.Scale(asset.realBoundingDiag, Eigen::Vector3d::Zero());
                asset.hierarchy = CADHierarchy::buildFromMesh(metricMesh, cadName);

                std::string symPath = cadDir + "/" + cadName + "_symmetry.json";
                if (QFile::exists(QString::fromStdString(symPath))) {
                    asset.hierarchy.symmetry = SymmetryDefinition::loadFromJson(symPath);
                }
                if (!QFile::exists(QString::fromStdString(triNpz))) {
                    asset.hierarchy.exportToFiles(cadDir, cadName);
                }
            }
            return asset;
        }
    }

    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    if (!open3d::io::ReadTriangleMesh(path, *mesh) || mesh->vertices_.empty()) {
        std::cerr << "Failed to read mesh or mesh is empty: " << path << "\n";
        return asset; // invalid asset, caller checks asset.valid()
    }

    double rawDiag = (mesh->GetMaxBound() - mesh->GetMinBound()).norm();
    if (rawDiag < 1e-9) {
        std::cerr << "Mesh has zero bounding diagonal.\n";
        return asset;
    }

    // Auto-detect unit if no .unit sidecar was provided:
    // If raw bounding diagonal > 10.0 (e.g. 50-200mm for household parts, 1000mm for tables),
    // the mesh was authored in millimeters! In meters it would be a 150m skyscraper.
    if (!unitExplicitlyFound) {
        if (rawDiag > 10.0) {
            unitScale = 0.001; // Auto-detect millimeters
            std::cerr << "CADPreprocessor: [AUTO-UNIT] Không tìm thấy file .unit và đường chéo CAD = "
                      << rawDiag << " > 10.0 -- Tự động xác định đơn vị là MILLIMETERS (mm, scale = 0.001 m/unit).\n";
            std::ofstream outUnit(path + ".unit");
            if (outUnit.is_open()) outUnit << "mm\n";
        } else {
            unitScale = 1.0; // meters
        }
    }

    asset.sourcePath = path;
    asset.sourceUnitScale = unitScale;

    Eigen::Vector3d rawCentroid = mesh->GetCenter();
    mesh->ComputeVertexNormals();

    mesh->Translate(-rawCentroid);
    mesh->Scale(1.0 / rawDiag, Eigen::Vector3d::Zero());

    asset.normalizedMesh = mesh;
    asset.centroid = Eigen::Vector3d::Zero(); // after normalize, centroid is at origin
    asset.boundingDiag = 1.0;                  // after scale, diag is 1
    asset.realBoundingDiag = rawDiag * unitScale; // physical size in METERS, needed to
                                                   // rescale back to metric for Phase 2
                                                   // matching against RGB-D

    // Sanity heads-up (not a hard failure -- some legitimate parts really
    // are this small/large): a wrong .unit sidecar most often shows up as
    // an implausible physical size for a hand-scale industrial part.
    if (asset.realBoundingDiag < 0.001 || asset.realBoundingDiag > 50.0) {
        std::cerr << "CADPreprocessor: " << path << " has realBoundingDiag = "
                  << asset.realBoundingDiag << " m after unit conversion -- "
                     "double check the .unit sidecar if this looks wrong.\n";
    }

    // Default object frame
    asset.frame.origin = asset.centroid;
    computePCAFrame(asset);

    // --- Poisson-disk sample
    auto sampledCloud = mesh->SamplePointsPoissonDisk(targetPointCount);
    if (!sampledCloud || sampledCloud->points_.empty()) {
        std::cerr << "Failed to sample points.\n";
        return asset;
    }
    sampledCloud->EstimateNormals();

    asset.sampleRadius = neighborhoodRadius;
    computeLocalFeatures(asset, *sampledCloud, neighborhoodRadius);

    // Build and export Canonical CAD Hierarchy (Phase 0 & 1)
    open3d::geometry::TriangleMesh metricMesh = *mesh;
    metricMesh.Scale(asset.realBoundingDiag, Eigen::Vector3d::Zero());

    QFileInfo cadFi(QString::fromStdString(path));
    std::string cadDir = cadFi.dir().absolutePath().toStdString();
    std::string cadName = cadFi.baseName().toStdString();

    asset.hierarchy = CADHierarchy::buildFromMesh(metricMesh, cadName);
    std::string symPath = cadDir + "/" + cadName + "_symmetry.json";
    if (QFile::exists(QString::fromStdString(symPath))) {
        asset.hierarchy.symmetry = SymmetryDefinition::loadFromJson(symPath);
    }
    asset.hierarchy.exportToFiles(cadDir, cadName);

    // Cache the result
    if (!asset.saveToFile(cachePath)) {
        std::cerr << "Warning: Failed to cache CADAsset to " << cachePath << "\n";
    }

    return asset;
}

void CADPreprocessor::computeLocalFeatures(CADAsset& asset,
                                            const open3d::geometry::PointCloud& sampledCloud,
                                            double radius) {
    open3d::geometry::KDTreeFlann kdtree(sampledCloud);
    asset.sampledFeatures.clear();
    asset.sampledFeatures.reserve(sampledCloud.points_.size());

    for (size_t i = 0; i < sampledCloud.points_.size(); ++i) {
        CadPointFeature f;
        f.position = sampledCloud.points_[i];
        f.normal = sampledCloud.normals_[i];

        std::vector<int> idx;
        std::vector<double> distSq;
        int found = kdtree.SearchRadius(f.position, radius, idx, distSq);
        f.neighborCount = found;

        if (found >= 3) {
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            for (int id : idx) mean += sampledCloud.points_[id];
            mean /= static_cast<double>(found);

            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (int id : idx) {
                Eigen::Vector3d d = sampledCloud.points_[id] - mean;
                cov += d * d.transpose();
            }
            cov /= static_cast<double>(found);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
            Eigen::Vector3d eigenvalues = solver.eigenvalues(); // ascending order
            double sum = eigenvalues.sum();
            f.curvature = sum > 1e-12 ? eigenvalues(0) / sum : 0.0;
        }

        asset.sampledFeatures.push_back(f);
    }

    double sumNeighbors = 0.0;
    int countUnder3 = 0;
    for (const auto& f : asset.sampledFeatures) {
        sumNeighbors += f.neighborCount;
        if (f.neighborCount < 3) countUnder3++;
    }
    double meanNeighbors = asset.sampledFeatures.empty() ? 0 : sumNeighbors / asset.sampledFeatures.size();
    double fracUnder3 = asset.sampledFeatures.empty() ? 0 : static_cast<double>(countUnder3) / asset.sampledFeatures.size();
    std::cout << "[CADPreprocessor] Points: " << asset.sampledFeatures.size() 
              << ", Radius: " << radius 
              << " | Mean neighbors: " << meanNeighbors 
              << ", Frac < 3: " << fracUnder3 << "\n";
}

void CADPreprocessor::computePCAFrame(CADAsset& asset) {
    if (!asset.normalizedMesh || asset.normalizedMesh->vertices_.empty()) {
        asset.frame.axes = Eigen::Matrix3d::Identity();
        return;
    }
    
    // WARNING (Phase 1): PCA frames for highly symmetric or elongated objects
    // (like Racks, where X and Z eigenvalues are nearly identical because
    // width ~ depth) are unstable. A small change in triangulation can flip
    // the X and Z axes 90 degrees or reverse their direction. If Phase 4
    // uses this frame to initialize ICP, a flipped axis will result in 90-degree
    // rotation errors. Do not rely blindly on PCA axes for orientation prior
    // on such objects (consider using a functional axis or gravity vector).
    
    // Compute covariance matrix of mesh vertices around their OWN mean.
    // NOTE: does not assume the mesh is already centered at the origin --
    // process() happens to call this after centering, but computing mean
    // explicitly here makes the function correct regardless of caller order,
    // instead of silently relying on that external precondition.
    const auto& vertices = asset.normalizedMesh->vertices_;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& v : vertices) mean += v;
    mean /= static_cast<double>(vertices.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& v : vertices) {
        Eigen::Vector3d d = v - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(vertices.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d eigenVectors = solver.eigenvectors();

    // Set frame axes aligned with principal components
    // x = major axis, y = medium, z = minor
    asset.frame.axes.col(0) = eigenVectors.col(2);
    asset.frame.axes.col(1) = eigenVectors.col(1);
    asset.frame.axes.col(2) = eigenVectors.col(0);

    // Ensure right-handed coordinate system
    if (asset.frame.axes.determinant() < 0) {
        asset.frame.axes.col(2) = -asset.frame.axes.col(2);
    }
}