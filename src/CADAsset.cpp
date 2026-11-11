#include "CADAsset.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <sys/stat.h>

namespace {
    // Returns mtime (seconds since epoch) of a file, or 0 if it can't be stat'd.
    int64_t fileMtime(const std::string& path) {
        struct stat st;
        if (stat(path.c_str(), &st) != 0) return 0;
        return static_cast<int64_t>(st.st_mtime);
    }
}

namespace {
    // Bumped to '04': format now also stores sourceUnitScale (the mm/cm/m/in
    // -> meters factor applied to compute realBoundingDiag). Older caches
    // ('01'-'03') are rejected below and rebuilt rather than misread --
    // reading a '03' cache as '04' would misinterpret trailing bytes as
    // sourceUnitScale and silently corrupt every downstream Phase-2 pose.
    const char MAGIC[8] = {'C', 'A', 'D', 'A', 'S', 'T', '0', '4'};
}

template <typename T>
void writePOD(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void readPOD(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
void writeVector(std::ofstream& out, const std::vector<T>& vec) {
    size_t size = vec.size();
    writePOD(out, size);
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
    }
}

template <typename T>
void readVector(std::ifstream& in, std::vector<T>& vec) {
    size_t size = 0;
    readPOD(in, size);
    vec.resize(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    }
}

bool CADAsset::saveToFile(const std::string& filepath) const {
    if (!valid()) return false;

    // Snapshot the current mtime of sourcePath at save time (rather than
    // trusting whatever sourceMtime already holds), so the cache always
    // reflects the file state it was actually built from.
    int64_t mtimeToWrite = fileMtime(sourcePath);

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing CADAsset.\n";
        return false;
    }

    // Write magic
    out.write(MAGIC, sizeof(MAGIC));

    // Write sourcePath
    size_t pathLen = sourcePath.length();
    writePOD(out, pathLen);
    if (pathLen > 0) {
        out.write(sourcePath.data(), pathLen);
    }

    // Write mtime snapshot of sourcePath, so loadFromFile() can tell if the
    // original CAD file was edited after this cache was built.
    writePOD(out, mtimeToWrite);

    // Write primitive fields
    writePOD(out, frame);
    writePOD(out, centroid);
    writePOD(out, boundingDiag);
    writePOD(out, sourceUnitScale);
    writePOD(out, realBoundingDiag);
    writePOD(out, sampleRadius);

    // Write features
    writeVector(out, sampledFeatures);

    // Write mesh components
    writeVector(out, normalizedMesh->vertices_);
    writeVector(out, normalizedMesh->triangles_);
    writeVector(out, normalizedMesh->vertex_normals_);
    writeVector(out, normalizedMesh->vertex_colors_);

    return out.good();
}

bool CADAsset::loadFromFile(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[8];
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
        std::cerr << "Invalid CADAsset file format: " << filepath << "\n";
        return false;
    }

    size_t pathLen = 0;
    readPOD(in, pathLen);
    if (pathLen > 0) {
        sourcePath.resize(pathLen);
        in.read(&sourcePath[0], pathLen);
    } else {
        sourcePath.clear();
    }

    int64_t storedMtime = 0;
    readPOD(in, storedMtime);
    sourceMtime = storedMtime;

    // Reject a cache whose source CAD file has since been modified. A
    // mismatch here means the mesh/features below no longer correspond to
    // what's currently on disk at sourcePath -- silently trusting it would
    // let a stale cache poison every downstream phase (2-7) without any
    // visible symptom.
    if (!sourcePath.empty()) {
        int64_t currentMtime = fileMtime(sourcePath);
        if (currentMtime != 0 && currentMtime != storedMtime) {
            std::cerr << "Stale CADAsset cache (source file changed since caching): "
                      << filepath << "\n";
            return false;
        }
    }

    readPOD(in, frame);
    readPOD(in, centroid);
    readPOD(in, boundingDiag);
    readPOD(in, sourceUnitScale);
    readPOD(in, realBoundingDiag);
    readPOD(in, sampleRadius);

    readVector(in, sampledFeatures);

    normalizedMesh = std::make_shared<open3d::geometry::TriangleMesh>();
    readVector(in, normalizedMesh->vertices_);
    readVector(in, normalizedMesh->triangles_);
    readVector(in, normalizedMesh->vertex_normals_);
    readVector(in, normalizedMesh->vertex_colors_);

    return in.good();
}