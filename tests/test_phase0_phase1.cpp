#include "../src/CADPreprocessor.h"
#include "../src/CADHierarchy.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <QFileInfo>
#include <QFile>
#include <QDir>

int main(int argc, char** argv) {
    std::cout << "============================================================\n";
    std::cout << " TEST PHASE 0 & PHASE 1: Canonical CAD Hierarchy & Symmetry\n";
    std::cout << "============================================================\n";

    std::string cadPath = (argc > 1) ? argv[1] : "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO/TABLE-DEMO.stl";
    if (!QFile::exists(QString::fromStdString(cadPath))) {
        cadPath = "/home/huyen1905/TMCAD/CAD_DATA/TABLE-DEMO.stl";
    }
    if (!QFile::exists(QString::fromStdString(cadPath))) {
        cadPath = "/home/huyen1905/TMCAD/CAD_DATA/ASYMMETRIC/ASYMMETRIC.stl";
    }
    if (!QFile::exists(QString::fromStdString(cadPath))) {
        cadPath = "/home/huyen1905/TMCAD/CAD_DATA/ASYMMETRIC.stl";
    }

    std::cout << "Loading CAD: " << cadPath << "\n";
    CADAsset asset = CADPreprocessor::process(cadPath);

    assert(asset.valid());
    std::cout << "[PASS] CAD Asset processed successfully (realBoundingDiag = " 
              << asset.realBoundingDiag << " m)\n";

    const auto& hier = asset.hierarchy;
    assert(hier.valid());
    std::cout << "[PASS] CADHierarchy is valid.\n";

    // 1. Triangle primitives
    size_t numTris = hier.triangles.size();
    std::cout << "  - Triangles count: " << numTris << "\n";
    assert(numTris > 0);

    const auto& t0 = hier.triangles[0];
    auto feat13 = t0.toFeatureVector13();
    assert(feat13.size() == 13);
    assert(t0.area > 0.0);
    assert(std::abs(t0.normal.norm() - 1.0) < 1e-3);
    std::cout << "[PASS] Triangle 0 feature vector (13D) validated. Area = " << t0.area << " m^2\n";

    // 2. Adjacency Graph
    int numEdges = hier.adjacency.numEdges;
    std::cout << "  - Adjacency directed edges: " << numEdges << "\n";
    assert(numEdges > 0);
    assert(hier.adjacency.edgeIndexU.size() == static_cast<size_t>(numEdges));
    assert(hier.adjacency.edgeAttr.size() == static_cast<size_t>(numEdges * 6));
    std::cout << "[PASS] Triangle Adjacency Graph validated.\n";

    // 3. Regions
    int numRegions = hier.regions.numRegions;
    std::cout << "  - Segmented Regions count: " << numRegions << "\n";
    assert(numRegions > 0);
    assert(hier.regions.regionMeanNormals.size() == static_cast<size_t>(numRegions));
    std::cout << "[PASS] Region Segmentation validated.\n";

    // 4. Local Patches
    assert(hier.patches.patchSizeK == 32);
    assert(hier.patches.patchIndices.size() == numTris * 32);
    std::cout << "[PASS] Local Patches (K=32) validated.\n";

    // 5. Phase 0 Symmetry
    std::cout << "  - Symmetry class: " << hier.symmetry.symmetryType << "\n";
    std::vector<Eigen::Vector3d> dummyPoints = {
        {0.1, 0.2, 0.3}, {-0.1, -0.2, 0.3}, {0.4, -0.1, 0.0}
    };
    Eigen::Matrix4d T_gt = Eigen::Matrix4d::Identity();
    T_gt(0, 3) = 0.5;
    T_gt(1, 3) = -0.2;
    T_gt(2, 3) = 1.2;

    // Test ADD on identical pose
    double addZero = hier.symmetry.computeADD(T_gt, T_gt, dummyPoints);
    assert(addZero < 1e-9);

    if (hier.symmetry.isSymmetric) {
        // Test with 180 deg yaw rotation around Z
        Eigen::Matrix4d T_rot180 = T_gt * hier.symmetry.symmetryTransforms[1];
        double addDiff = hier.symmetry.computeADD(T_rot180, T_gt, dummyPoints);
        double addsZero = hier.symmetry.computeADDS(T_rot180, T_gt, dummyPoints);
        std::cout << "  - ADD(T_rot180, T_gt)  = " << addDiff * 1000.0 << " mm (shows difference)\n";
        std::cout << "  - ADD-S(T_rot180, T_gt) = " << addsZero * 1000.0 << " mm (shows invariance)\n";
        assert(addsZero < 1e-7);
    }
    std::cout << "[PASS] Symmetry metrics (ADD / ADD-S) validated.\n";

    // 6. Check exported canonical files
    QFileInfo fi(QString::fromStdString(cadPath));
    QString baseDir = fi.dir().absolutePath();
    QString baseName = fi.baseName();

    QString plyFile = QString("%1/%2_mesh.ply").arg(baseDir, baseName);
    QString triNpz = QString("%1/%2_triangles.npz").arg(baseDir, baseName);
    QString adjNpz = QString("%1/%2_adjacency.npz").arg(baseDir, baseName);
    QString patNpz = QString("%1/%2_patches.npz").arg(baseDir, baseName);
    QString regNpz = QString("%1/%2_regions.npz").arg(baseDir, baseName);
    QString symJson = QString("%1/%2_symmetry.json").arg(baseDir, baseName);

    assert(QFile::exists(plyFile));
    assert(QFile::exists(triNpz));
    assert(QFile::exists(adjNpz));
    assert(QFile::exists(patNpz));
    assert(QFile::exists(regNpz));
    assert(QFile::exists(symJson));

    std::cout << "[PASS] All canonical files exported successfully:\n"
              << "       " << plyFile.toStdString() << "\n"
              << "       " << triNpz.toStdString() << "\n"
              << "       " << adjNpz.toStdString() << "\n"
              << "       " << patNpz.toStdString() << "\n"
              << "       " << regNpz.toStdString() << "\n"
              << "       " << symJson.toStdString() << "\n";

    std::cout << "\n>>> ALL PHASE 0 & PHASE 1 TESTS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
