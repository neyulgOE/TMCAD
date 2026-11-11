#pragma once
#include "CADAsset.h"
#include <string>

class CADPreprocessor {
public:
    // path: original CAD file (mesh, NOT pre-normalized - this function handles it).
    // targetPointCount: number of Poisson-disk samples desired.
    // neighborhoodRadius: radius (as a fraction of the normalized bounding
    //   diagonal, which is always 1.0 after normalize) used for curvature /
    //   The defaults (5000 points, 0.06 radius) are calibrated for
    //   a ~1.65m diagonal Rack (Phase 1). They ensure dense enough coverage
    //   and enough neighborhood overlap to compute stable curvature estimates (enough neighbors,
    //   not so large it blurs distinct features together) before Phase 2
    //   relies on it for descriptor matching.
    //
    // Reads a ".unit" sidecar next to path (see readSourceUnitScale) to
    // convert the raw file's coordinates to meters before computing
    // CADAsset::realBoundingDiag -- required for Phase 2's metric matching
    // to be correct across CAD files authored in different units.
    static CADAsset process(const std::string& path,
                             size_t targetPointCount = 5000,
                             double neighborhoodRadius = 0.06);


    // Reads the unit sidecar next to a CAD file (path + ".unit") and
    // returns the scale factor to multiply the file's raw coordinates by
    // to get meters. Recognized contents (case-insensitive, first
    // whitespace-delimited token): "m"/"meter(s)", "cm"/"centimeter(s)",
    // "mm"/"millimeter(s)", "in"/"inch"/"inches".
    //
    // A missing sidecar or unrecognized content logs a warning and
    // defaults to 1.0 (assumes the file is already in meters) -- getting
    // this wrong silently makes every downstream Phase-2 pose off by the
    // corresponding factor (e.g. 1000x for an mm file mistaken for
    // meters), so create "<file>.unit" containing e.g. "mm" for any CAD
    // file not authored in meters.
    static double readSourceUnitScale(const std::string& path);

private:
    // Fills asset.sampledFeatures from sampledCloud: per-point normal
    // (copied from the cloud) plus curvature/neighborCount estimated via
    // KD-tree radius search + covariance eigenanalysis.
    static void computeLocalFeatures(CADAsset& asset,
                                      const open3d::geometry::PointCloud& sampledCloud,
                                      double radius);
    static void computePCAFrame(CADAsset& asset);
};