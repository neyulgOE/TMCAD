#pragma once

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// ============================================================================
// PHASE 0: Symmetry Definition & Robust Metrics
// ============================================================================
struct SymmetryDefinition {
    std::string cadName;
    bool isSymmetric = false;
    std::string symmetryType = "None"; // "None", "Discrete_C2_Z", "Discrete_C4_Z", "Continuous_Z"
    
    // Group of 4x4 transformation matrices in object space that leave
    // the physical appearance and geometry invariant.
    std::vector<Eigen::Matrix4d> symmetryTransforms;

    static SymmetryDefinition createDefault(const std::string& name);
    static SymmetryDefinition loadFromJson(const std::string& jsonPath);
    bool saveToJson(const std::string& jsonPath) const;

    // Metrics
    double computeADD(const Eigen::Matrix4d& T_pred,
                      const Eigen::Matrix4d& T_gt,
                      const std::vector<Eigen::Vector3d>& cadPointsMetric) const;

    double computeADDS(const Eigen::Matrix4d& T_pred,
                       const Eigen::Matrix4d& T_gt,
                       const std::vector<Eigen::Vector3d>& cadPointsMetric) const;

    double computeRotErrorDeg(const Eigen::Matrix3d& R_pred,
                              const Eigen::Matrix3d& R_gt) const;

    double computeSymmetricRotErrorDeg(const Eigen::Matrix3d& R_pred,
                                       const Eigen::Matrix3d& R_gt) const;
};

// ============================================================================
// PHASE 1: Canonical CAD Hierarchy Components
// ============================================================================

// 1.1 Triangle Primitive: Xi in R^13
struct TrianglePrimitive {
    int id = 0;
    Eigen::Vector3i vertexIndices{0, 0, 0};
    Eigen::Vector3d v0{0, 0, 0};
    Eigen::Vector3d v1{0, 0, 0};
    Eigen::Vector3d v2{0, 0, 0};
    Eigen::Vector3d centroid{0, 0, 0};
    Eigen::Vector3d normal{0, 0, 1};
    double area = 0.0;
    Eigen::Vector3d edgeLengths{0, 0, 0}; // (l0, l1, l2)
    Eigen::Vector3d angles{0, 0, 0};      // (a0, a1, a2) in radians
    int regionId = -1;

    // Vector [cx, cy, cz, nx, ny, nz, area, l0, l1, l2, a0, a1, a2] (dim=13)
    std::vector<float> toFeatureVector13() const;
};

// 1.2 Triangle Adjacency Graph: edge_index (2, E) & edge_attr (E, 6)
struct TriangleAdjacencyGraph {
    int numTriangles = 0;
    int numEdges = 0;
    std::vector<int> edgeIndexU; // Source triangle IDs
    std::vector<int> edgeIndexV; // Target triangle IDs
    // Edge features: [delta_cx, delta_cy, delta_cz, shared_edge_len, dihedral_angle, normal_dot]
    std::vector<float> edgeAttr; 
};

// 1.3 Local Patches: K-ring neighborhoods
struct LocalPatchBundle {
    int patchSizeK = 32;
    std::vector<int> patchIndices; // Shape (M * K)
    std::vector<int> patchHops;    // Shape (M * K) graph distance
};

// 1.4 Region Segmentation: Grouping triangles by dihedral angle
struct RegionSegmentation {
    int numRegions = 0;
    std::vector<int> triangleRegionIds; // Shape (M)
    std::vector<Eigen::Vector3d> regionMeanNormals;
    std::vector<double> regionAreas;
    std::vector<int> regionSizes;
};

// ============================================================================
// Complete Canonical CAD Hierarchy Bundle
// ============================================================================
class CADHierarchy {
public:
    std::vector<TrianglePrimitive> triangles;
    TriangleAdjacencyGraph adjacency;
    LocalPatchBundle patches;
    RegionSegmentation regions;
    SymmetryDefinition symmetry;

    bool valid() const { return !triangles.empty(); }

    // Build the full 4-tier hierarchy from a metric TriangleMesh
    static CADHierarchy buildFromMesh(const open3d::geometry::TriangleMesh& mesh,
                                      const std::string& cadName = "CAD",
                                      double dihedralThreshDeg = 15.0,
                                      int patchSize = 32);

    // Exports all canonical files for Phase 4 (Python / Deep Learning):
    //   <outputDir>/<cadName>_mesh.ply
    //   <outputDir>/<cadName>_triangles.npz
    //   <outputDir>/<cadName>_adjacency.npz
    //   <outputDir>/<cadName>_patches.npz
    //   <outputDir>/<cadName>_regions.npz
    //   <outputDir>/<cadName>_symmetry.json
    bool exportToFiles(const std::string& outputDir, const std::string& cadName) const;
};
