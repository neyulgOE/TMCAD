#include "CADHierarchy.h"
#include <open3d/t/io/NumpyIO.h>
#include <open3d/core/Tensor.h>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFileInfo>
#include <QDir>

#include <iostream>
#include <queue>
#include <cmath>
#include <algorithm>

namespace {
inline uint64_t makeEdgeKey(int u, int v) {
    uint32_t a = static_cast<uint32_t>(std::min(u, v));
    uint32_t b = static_cast<uint32_t>(std::max(u, v));
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

inline double clampVal(double val, double minVal, double maxVal) {
    return std::max(minVal, std::min(maxVal, val));
}
} // namespace

// ============================================================================
// TrianglePrimitive
// ============================================================================
std::vector<float> TrianglePrimitive::toFeatureVector13() const {
    std::vector<float> feat(13);
    feat[0]  = static_cast<float>(centroid.x());
    feat[1]  = static_cast<float>(centroid.y());
    feat[2]  = static_cast<float>(centroid.z());
    feat[3]  = static_cast<float>(normal.x());
    feat[4]  = static_cast<float>(normal.y());
    feat[5]  = static_cast<float>(normal.z());
    feat[6]  = static_cast<float>(area);
    feat[7]  = static_cast<float>(edgeLengths.x());
    feat[8]  = static_cast<float>(edgeLengths.y());
    feat[9]  = static_cast<float>(edgeLengths.z());
    feat[10] = static_cast<float>(angles.x());
    feat[11] = static_cast<float>(angles.y());
    feat[12] = static_cast<float>(angles.z());
    return feat;
}

// ============================================================================
// SymmetryDefinition
// ============================================================================
SymmetryDefinition SymmetryDefinition::createDefault(const std::string& name) {
    SymmetryDefinition sym;
    sym.cadName = name;

    // Check known symmetric models (e.g. TABLE-DEMO has 180-deg yaw symmetry around Z)
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

    if (lowerName.find("table") != std::string::npos) {
        sym.isSymmetric = true;
        sym.symmetryType = "Discrete_C2_Z";

        // Identity
        sym.symmetryTransforms.push_back(Eigen::Matrix4d::Identity());

        // 180 deg around Z
        Eigen::Matrix4d rot180 = Eigen::Matrix4d::Identity();
        rot180(0, 0) = -1.0;
        rot180(1, 1) = -1.0;
        rot180(2, 2) =  1.0;
        sym.symmetryTransforms.push_back(rot180);
    } else {
        sym.isSymmetric = false;
        sym.symmetryType = "None";
        sym.symmetryTransforms.push_back(Eigen::Matrix4d::Identity());
    }

    return sym;
}

SymmetryDefinition SymmetryDefinition::loadFromJson(const std::string& jsonPath) {
    SymmetryDefinition sym;
    QFile file(QString::fromStdString(jsonPath));
    if (!file.open(QIODevice::ReadOnly)) {
        sym = createDefault(QFileInfo(file).baseName().toStdString());
        return sym;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    sym.cadName = root["cad_name"].toString().toStdString();
    sym.isSymmetric = root["is_symmetric"].toBool();
    sym.symmetryType = root["symmetry_type"].toString("None").toStdString();

    QJsonArray xformsArray = root["symmetry_transforms"].toArray();
    sym.symmetryTransforms.clear();

    for (const auto& xformVal : xformsArray) {
        QJsonArray matArray = xformVal.toArray();
        if (matArray.size() == 16) {
            Eigen::Matrix4d T;
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    T(r, c) = matArray[r * 4 + c].toDouble();
                }
            }
            sym.symmetryTransforms.push_back(T);
        }
    }

    if (sym.symmetryTransforms.empty()) {
        sym.symmetryTransforms.push_back(Eigen::Matrix4d::Identity());
    }

    return sym;
}

bool SymmetryDefinition::saveToJson(const std::string& jsonPath) const {
    QJsonObject root;
    root["cad_name"] = QString::fromStdString(cadName);
    root["is_symmetric"] = isSymmetric;
    root["symmetry_type"] = QString::fromStdString(symmetryType);

    QJsonArray xformsArray;
    for (const auto& T : symmetryTransforms) {
        QJsonArray matArray;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                matArray.append(T(r, c));
            }
        }
        xformsArray.append(matArray);
    }
    root["symmetry_transforms"] = xformsArray;

    QFile file(QString::fromStdString(jsonPath));
    if (!file.open(QIODevice::WriteOnly)) {
        std::cerr << "Failed to open " << jsonPath << " for writing symmetry config.\n";
        return false;
    }

    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    return true;
}

double SymmetryDefinition::computeADD(const Eigen::Matrix4d& T_pred,
                                      const Eigen::Matrix4d& T_gt,
                                      const std::vector<Eigen::Vector3d>& cadPointsMetric) const {
    if (cadPointsMetric.empty()) return 0.0;

    Eigen::Matrix3d R_pred = T_pred.block<3, 3>(0, 0);
    Eigen::Vector3d t_pred = T_pred.block<3, 1>(0, 3);
    Eigen::Matrix3d R_gt   = T_gt.block<3, 3>(0, 0);
    Eigen::Vector3d t_gt   = T_gt.block<3, 1>(0, 3);

    double sumDist = 0.0;
    for (const auto& p : cadPointsMetric) {
        Eigen::Vector3d p_pred = R_pred * p + t_pred;
        Eigen::Vector3d p_gt   = R_gt * p + t_gt;
        sumDist += (p_pred - p_gt).norm();
    }
    return sumDist / static_cast<double>(cadPointsMetric.size());
}

double SymmetryDefinition::computeADDS(const Eigen::Matrix4d& T_pred,
                                       const Eigen::Matrix4d& T_gt,
                                       const std::vector<Eigen::Vector3d>& cadPointsMetric) const {
    if (cadPointsMetric.empty()) return 0.0;

    if (!isSymmetric || symmetryTransforms.size() <= 1) {
        return computeADD(T_pred, T_gt, cadPointsMetric);
    }

    // For discrete symmetry group G = {S_k}, min ADD over all equivalent poses
    double minAdd = 1e9;
    for (const auto& S : symmetryTransforms) {
        Eigen::Matrix4d T_gt_sym = T_gt * S;
        double addVal = computeADD(T_pred, T_gt_sym, cadPointsMetric);
        if (addVal < minAdd) {
            minAdd = addVal;
        }
    }
    return minAdd;
}

double SymmetryDefinition::computeRotErrorDeg(const Eigen::Matrix3d& R_pred,
                                              const Eigen::Matrix3d& R_gt) const {
    Eigen::Matrix3d R_diff = R_pred * R_gt.transpose();
    double tr = R_diff.trace();
    double cosTheta = clampVal((tr - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(cosTheta) * 180.0 / M_PI;
}

double SymmetryDefinition::computeSymmetricRotErrorDeg(const Eigen::Matrix3d& R_pred,
                                                       const Eigen::Matrix3d& R_gt) const {
    if (!isSymmetric || symmetryTransforms.size() <= 1) {
        return computeRotErrorDeg(R_pred, R_gt);
    }

    double minDeg = 1e9;
    for (const auto& S : symmetryTransforms) {
        Eigen::Matrix3d R_s = S.block<3, 3>(0, 0);
        Eigen::Matrix3d R_gt_sym = R_gt * R_s;
        double deg = computeRotErrorDeg(R_pred, R_gt_sym);
        if (deg < minDeg) {
            minDeg = deg;
        }
    }
    return minDeg;
}

// ============================================================================
// CADHierarchy
// ============================================================================
CADHierarchy CADHierarchy::buildFromMesh(const open3d::geometry::TriangleMesh& mesh,
                                         const std::string& cadName,
                                         double dihedralThreshDeg,
                                         int patchSize) {
    CADHierarchy hierarchy;
    hierarchy.symmetry = SymmetryDefinition::createDefault(cadName);

    size_t numTris = mesh.triangles_.size();
    if (numTris == 0) return hierarchy;

    hierarchy.triangles.resize(numTris);

    // Compute normals if missing
    open3d::geometry::TriangleMesh processedMesh = mesh;
    if (processedMesh.triangle_normals_.empty()) {
        processedMesh.ComputeTriangleNormals();
    }
    if (processedMesh.vertex_normals_.empty()) {
        processedMesh.ComputeVertexNormals();
    }

    const auto& verts = processedMesh.vertices_;
    const auto& tris  = processedMesh.triangles_;
    const auto& tNormals = processedMesh.triangle_normals_;

    // ── 1. Triangle Primitives ──────────────────────────────────────────────
    for (size_t i = 0; i < numTris; ++i) {
        TrianglePrimitive& p = hierarchy.triangles[i];
        p.id = static_cast<int>(i);
        p.vertexIndices = tris[i];
        p.v0 = verts[p.vertexIndices(0)];
        p.v1 = verts[p.vertexIndices(1)];
        p.v2 = verts[p.vertexIndices(2)];

        p.centroid = (p.v0 + p.v1 + p.v2) / 3.0;

        Eigen::Vector3d e0 = p.v1 - p.v0;
        Eigen::Vector3d e1 = p.v2 - p.v1;
        Eigen::Vector3d e2 = p.v0 - p.v2;

        p.edgeLengths.x() = e0.norm();
        p.edgeLengths.y() = e1.norm();
        p.edgeLengths.z() = e2.norm();

        Eigen::Vector3d crossProd = e0.cross(-e2);
        double crossNorm = crossProd.norm();
        p.area = 0.5 * crossNorm;

        if (i < tNormals.size() && tNormals[i].norm() > 1e-6) {
            p.normal = tNormals[i].normalized();
        } else if (crossNorm > 1e-9) {
            p.normal = crossProd / crossNorm;
        } else {
            p.normal = Eigen::Vector3d(0, 0, 1);
        }

        // Internal angles (radians)
        double l0 = std::max(1e-9, p.edgeLengths.x());
        double l1 = std::max(1e-9, p.edgeLengths.y());
        double l2 = std::max(1e-9, p.edgeLengths.z());

        double cosA0 = clampVal(e0.dot(-e2) / (l0 * l2), -1.0, 1.0);
        double cosA1 = clampVal(e1.dot(-e0) / (l1 * l0), -1.0, 1.0);
        p.angles.x() = std::acos(cosA0);
        p.angles.y() = std::acos(cosA1);
        p.angles.z() = std::max(0.0, M_PI - p.angles.x() - p.angles.y());
    }

    // ── 2. Triangle Adjacency Graph ─────────────────────────────────────────
    std::unordered_map<uint64_t, std::vector<int>> edgeToTriangles;
    edgeToTriangles.reserve(numTris * 3);

    for (size_t i = 0; i < numTris; ++i) {
        int idx0 = tris[i](0);
        int idx1 = tris[i](1);
        int idx2 = tris[i](2);

        edgeToTriangles[makeEdgeKey(idx0, idx1)].push_back(static_cast<int>(i));
        edgeToTriangles[makeEdgeKey(idx1, idx2)].push_back(static_cast<int>(i));
        edgeToTriangles[makeEdgeKey(idx2, idx0)].push_back(static_cast<int>(i));
    }

    // Graph adjacency list for BFS: triangle -> list of (neighbor_triangle, dihedral_deg)
    std::vector<std::vector<std::pair<int, double>>> adjList(numTris);

    hierarchy.adjacency.numTriangles = static_cast<int>(numTris);
    hierarchy.adjacency.edgeIndexU.clear();
    hierarchy.adjacency.edgeIndexV.clear();
    hierarchy.adjacency.edgeAttr.clear();

    for (const auto& pair : edgeToTriangles) {
        const auto& tList = pair.second;
        if (tList.size() == 2) {
            int tA = tList[0];
            int tB = tList[1];

            const auto& triA = hierarchy.triangles[tA];
            const auto& triB = hierarchy.triangles[tB];

            Eigen::Vector3d deltaC = triB.centroid - triA.centroid;

            // Shared edge length
            uint32_t vIdx0 = static_cast<uint32_t>(pair.first >> 32);
            uint32_t vIdx1 = static_cast<uint32_t>(pair.first & 0xFFFFFFFF);
            double edgeLen = (verts[vIdx0] - verts[vIdx1]).norm();

            double dotNorm = clampVal(triA.normal.dot(triB.normal), -1.0, 1.0);
            double dihedralRad = std::acos(dotNorm);
            double dihedralDeg = dihedralRad * 180.0 / M_PI;

            adjList[tA].push_back({tB, dihedralDeg});
            adjList[tB].push_back({tA, dihedralDeg});

            // Edge A -> B
            hierarchy.adjacency.edgeIndexU.push_back(tA);
            hierarchy.adjacency.edgeIndexV.push_back(tB);
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(deltaC.x()));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(deltaC.y()));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(deltaC.z()));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(edgeLen));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(dihedralRad));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(dotNorm));

            // Edge B -> A
            hierarchy.adjacency.edgeIndexU.push_back(tB);
            hierarchy.adjacency.edgeIndexV.push_back(tA);
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(-deltaC.x()));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(-deltaC.y()));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(-deltaC.z()));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(edgeLen));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(dihedralRad));
            hierarchy.adjacency.edgeAttr.push_back(static_cast<float>(dotNorm));
        }
    }
    hierarchy.adjacency.numEdges = static_cast<int>(hierarchy.adjacency.edgeIndexU.size());

    // ── 3. Region Segmentation via Dihedral Angle BFS ─────────────────────────
    hierarchy.regions.triangleRegionIds.assign(numTris, -1);
    int currentRegionId = 0;

    for (size_t i = 0; i < numTris; ++i) {
        if (hierarchy.regions.triangleRegionIds[i] != -1) continue;

        std::queue<int> q;
        q.push(static_cast<int>(i));
        hierarchy.regions.triangleRegionIds[i] = currentRegionId;

        Eigen::Vector3d regionNormalSum(0, 0, 0);
        double regionAreaSum = 0.0;
        int regionTriCount = 0;

        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            regionNormalSum += hierarchy.triangles[curr].normal * hierarchy.triangles[curr].area;
            regionAreaSum   += hierarchy.triangles[curr].area;
            regionTriCount++;

            for (const auto& edge : adjList[curr]) {
                int next = edge.first;
                double angleDeg = edge.second;

                if (hierarchy.regions.triangleRegionIds[next] == -1 && angleDeg <= dihedralThreshDeg) {
                    hierarchy.regions.triangleRegionIds[next] = currentRegionId;
                    q.push(next);
                }
            }
        }

        if (regionAreaSum > 1e-9) {
            regionNormalSum.normalize();
        } else {
            regionNormalSum = Eigen::Vector3d(0, 0, 1);
        }

        hierarchy.regions.regionMeanNormals.push_back(regionNormalSum);
        hierarchy.regions.regionAreas.push_back(regionAreaSum);
        hierarchy.regions.regionSizes.push_back(regionTriCount);

        currentRegionId++;
    }
    hierarchy.regions.numRegions = currentRegionId;

    // Set regionId back to triangles
    for (size_t i = 0; i < numTris; ++i) {
        hierarchy.triangles[i].regionId = hierarchy.regions.triangleRegionIds[i];
    }

    // ── 4. Local Patches (K-ring BFS) ────────────────────────────────────────
    hierarchy.patches.patchSizeK = patchSize;
    hierarchy.patches.patchIndices.assign(numTris * patchSize, -1);
    hierarchy.patches.patchHops.assign(numTris * patchSize, -1);

    std::unordered_map<int, int> visitedHops;
    visitedHops.reserve(patchSize * 4);

    for (size_t i = 0; i < numTris; ++i) {
        visitedHops.clear();
        std::queue<int> q;

        int startIdx = static_cast<int>(i);
        q.push(startIdx);
        visitedHops[startIdx] = 0;

        int collected = 0;
        while (!q.empty() && collected < patchSize) {
            int curr = q.front();
            q.pop();

            int hop = visitedHops[curr];
            hierarchy.patches.patchIndices[i * patchSize + collected] = curr;
            hierarchy.patches.patchHops[i * patchSize + collected]    = hop;
            collected++;

            for (const auto& edge : adjList[curr]) {
                int next = edge.first;
                if (visitedHops.find(next) == visitedHops.end()) {
                    visitedHops[next] = hop + 1;
                    q.push(next);
                }
            }
        }
    }

    return hierarchy;
}

bool CADHierarchy::exportToFiles(const std::string& outputDir, const std::string& cadName) const {
    if (!valid()) return false;

    QDir dir(QString::fromStdString(outputDir));
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    std::string prefix = outputDir + "/" + cadName;

    // 1. Export mesh.ply
    open3d::geometry::TriangleMesh mesh;
    mesh.triangles_.reserve(triangles.size());
    mesh.vertices_.reserve(triangles.size() * 3);
    mesh.triangle_normals_.reserve(triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& t = triangles[i];
        int vIdx = static_cast<int>(mesh.vertices_.size());
        mesh.vertices_.push_back(t.v0);
        mesh.vertices_.push_back(t.v1);
        mesh.vertices_.push_back(t.v2);
        mesh.triangles_.push_back(Eigen::Vector3i(vIdx, vIdx + 1, vIdx + 2));
        mesh.triangle_normals_.push_back(t.normal);
    }
    mesh.ComputeVertexNormals();

    std::string plyPath = prefix + "_mesh.ply";
    open3d::io::WriteTriangleMesh(plyPath, mesh);

    // 2. Export symmetry.json
    std::string symPath = prefix + "_symmetry.json";
    symmetry.saveToJson(symPath);

    // 3. Export triangles.npz: features (M, 13), vertices (M, 3, 3), normals (M, 3)
    int64_t M = static_cast<int64_t>(triangles.size());
    std::vector<float> featData(M * 13);
    std::vector<float> vertData(M * 9);
    std::vector<float> normData(M * 3);

    for (int64_t i = 0; i < M; ++i) {
        auto f13 = triangles[i].toFeatureVector13();
        std::copy(f13.begin(), f13.end(), featData.begin() + i * 13);

        vertData[i * 9 + 0] = static_cast<float>(triangles[i].v0.x());
        vertData[i * 9 + 1] = static_cast<float>(triangles[i].v0.y());
        vertData[i * 9 + 2] = static_cast<float>(triangles[i].v0.z());
        vertData[i * 9 + 3] = static_cast<float>(triangles[i].v1.x());
        vertData[i * 9 + 4] = static_cast<float>(triangles[i].v1.y());
        vertData[i * 9 + 5] = static_cast<float>(triangles[i].v1.z());
        vertData[i * 9 + 6] = static_cast<float>(triangles[i].v2.x());
        vertData[i * 9 + 7] = static_cast<float>(triangles[i].v2.y());
        vertData[i * 9 + 8] = static_cast<float>(triangles[i].v2.z());

        normData[i * 3 + 0] = static_cast<float>(triangles[i].normal.x());
        normData[i * 3 + 1] = static_cast<float>(triangles[i].normal.y());
        normData[i * 3 + 2] = static_cast<float>(triangles[i].normal.z());
    }

    std::unordered_map<std::string, open3d::core::Tensor> triMap;
    triMap["features"] = open3d::core::Tensor(featData.data(), open3d::core::SizeVector{M, 13}, open3d::core::Dtype::Float32);
    triMap["vertices"] = open3d::core::Tensor(vertData.data(), open3d::core::SizeVector{M, 3, 3}, open3d::core::Dtype::Float32);
    triMap["normals"]  = open3d::core::Tensor(normData.data(), open3d::core::SizeVector{M, 3}, open3d::core::Dtype::Float32);

    open3d::t::io::WriteNpz(prefix + "_triangles.npz", triMap);

    // 4. Export adjacency.npz: edge_index (2, E), edge_attr (E, 6)
    int64_t E = static_cast<int64_t>(adjacency.numEdges);
    std::vector<int> edgeIndexData(2 * E);
    for (int64_t i = 0; i < E; ++i) {
        edgeIndexData[i]     = adjacency.edgeIndexU[i];
        edgeIndexData[E + i] = adjacency.edgeIndexV[i];
    }

    std::unordered_map<std::string, open3d::core::Tensor> adjMap;
    adjMap["edge_index"] = open3d::core::Tensor(edgeIndexData.data(), open3d::core::SizeVector{2, E}, open3d::core::Dtype::Int32);
    adjMap["edge_attr"]  = open3d::core::Tensor(adjacency.edgeAttr.data(), open3d::core::SizeVector{E, 6}, open3d::core::Dtype::Float32);

    open3d::t::io::WriteNpz(prefix + "_adjacency.npz", adjMap);

    // 5. Export patches.npz: patch_indices (M, K), patch_hops (M, K)
    int64_t K = static_cast<int64_t>(patches.patchSizeK);
    std::unordered_map<std::string, open3d::core::Tensor> patchMap;
    patchMap["patch_indices"] = open3d::core::Tensor(patches.patchIndices.data(), open3d::core::SizeVector{M, K}, open3d::core::Dtype::Int32);
    patchMap["patch_hops"]    = open3d::core::Tensor(patches.patchHops.data(), open3d::core::SizeVector{M, K}, open3d::core::Dtype::Int32);

    open3d::t::io::WriteNpz(prefix + "_patches.npz", patchMap);

    // 6. Export regions.npz: region_ids (M), region_normals (R, 3), region_areas (R)
    int64_t R = static_cast<int64_t>(regions.numRegions);
    std::vector<float> regNormData(R * 3);
    std::vector<float> regAreaData(R);

    for (int64_t i = 0; i < R; ++i) {
        regNormData[i * 3 + 0] = static_cast<float>(regions.regionMeanNormals[i].x());
        regNormData[i * 3 + 1] = static_cast<float>(regions.regionMeanNormals[i].y());
        regNormData[i * 3 + 2] = static_cast<float>(regions.regionMeanNormals[i].z());
        regAreaData[i]         = static_cast<float>(regions.regionAreas[i]);
    }

    std::unordered_map<std::string, open3d::core::Tensor> regMap;
    regMap["region_ids"]      = open3d::core::Tensor(regions.triangleRegionIds.data(), open3d::core::SizeVector{M}, open3d::core::Dtype::Int32);
    regMap["region_normals"]  = open3d::core::Tensor(regNormData.data(), open3d::core::SizeVector{R, 3}, open3d::core::Dtype::Float32);
    regMap["region_areas"]    = open3d::core::Tensor(regAreaData.data(), open3d::core::SizeVector{R}, open3d::core::Dtype::Float32);

    open3d::t::io::WriteNpz(prefix + "_regions.npz", regMap);

    return true;
}
