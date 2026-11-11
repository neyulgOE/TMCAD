#pragma once
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "CADHierarchy.h"

// Reference frame of the object, independent of display transform in viewport.
// origin: reference point used as pose origin (default = centroid).
struct ObjectFrame {
    Eigen::Vector3d origin{0, 0, 0};
    Eigen::Matrix3d axes = Eigen::Matrix3d::Identity(); // columns = x, y, z axes
};

// Local geometry feature for a sample point (for FPFH/SHOT in Phase 2,
// and as input for encoder in Phase 3).
struct CadPointFeature {
    Eigen::Vector3d position;
    Eigen::Vector3d normal;
    double curvature = 0.0;      // approximated via eigenvalue ratio of local covariance
    int neighborCount = 0;       // neighbor density in standard radius
};

// Preprocessed result of a CAD file - computed once, cached, shared for
// Phase 2 (classical descriptor) and Phase 3 (deep encoder input).
struct CADAsset {
    std::string sourcePath;
    int64_t sourceMtime = 0;     // mtime (seconds since epoch) of sourcePath when this
                                  // asset was built -- lets loadFromFile() detect a stale
                                  // cache if the original CAD file was edited afterwards.
    std::shared_ptr<open3d::geometry::TriangleMesh> normalizedMesh; // centered + scaled to unit diag
    ObjectFrame frame;
    Eigen::Vector3d centroid{0, 0, 0};
    double boundingDiag = 0.0;   // ALWAYS 1.0 after normalize -- unitless scale used for
                                  // sampleRadius and all downstream *shape* thresholds.

    // Scale factor (multiply raw CAD-file coordinates by this to get
    // meters) that CADPreprocessor::process() applied when computing
    // realBoundingDiag below. Comes from the CAD file's ".unit" sidecar
    // (see CADPreprocessor::readSourceUnitScale) -- defaults to 1.0
    // (assume meters) if no sidecar is present, with a loud warning at
    // process() time. Persisted here purely so loadFromFile() can detect
    // "the .unit sidecar changed since this cache was built" and force a
    // rebuild, the same way sourceMtime detects "the CAD file itself
    // changed" -- see CADPreprocessor::process().
    double sourceUnitScale = 1.0;

    // Real-world size of the CAD in METERS (rawBoundingDiag * sourceUnitScale
    // above) -- CAD authoring tools vary in what unit they save coordinates
    // in, so never assume this equals the raw file's bounding diagonal
    // without checking sourceUnitScale first.
    // This is what lets Phase 2 rescale the unit-normalized sampledFeatures
    // back to metric before matching against a real RGB-D point cloud --
    // normalizedMesh/sampledFeatures alone have NO physical scale.
    double realBoundingDiag = 0.0;

    std::vector<CadPointFeature> sampledFeatures; // from Poisson-disk sample, in the SAME
                                                   // unit-normalized space as normalizedMesh
                                                   // (i.e. NOT metric -- multiply by
                                                   // realBoundingDiag to get meters).
    double sampleRadius = 0.0;   // radius used for sampling, saved for reproduction
    CADHierarchy hierarchy;       // Canonical CAD hierarchy (Phase 0 & 1)

    bool valid() const { return normalizedMesh && !sampledFeatures.empty() && realBoundingDiag > 0.0; }

    // Serialization to a binary file
    bool saveToFile(const std::string& filepath) const;
    // Deserialization from a binary file
    bool loadFromFile(const std::string& filepath);
};