#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LUNG ABLATION DOSIMETRY — v12
==============================

CT + 99mTc-MAA SPECT/CT perfusion-weighted activity distribution
followed by beta dose-point-kernel convolution for:

    - Y-90
    - Lu-177

DENSITY-SCALED DPK + CT-DENSITY DOSE CORRECTION
-------------------------------------------------

The existing v14 Graves beta dose-point kernels are generated in water
(rho = 1.00 g/cm3) and physically scaled using the reference lung density
(rho = 0.26 g/cm3). That kernel construction is intentionally unchanged.

This version additionally incorporates the CT-HU-derived voxel-wise
density map from the previous dosimetry implementation. The final dose
is corrected voxel-by-voxel for local mass using:

    D_local = D_reference × rho_reference / rho_CT

Thus the spatial CT density variation affects the local dose-to-mass
conversion, while the underlying Graves DPK transport/radial scaling
remains the same as v14. This is not a full heterogeneous Monte Carlo
transport calculation.

REGISTRATION
------------

1. Load CT using true DICOM patient-coordinate geometry.
2. Load reconstructed SPECT.
3. Use reconstructed SPECT DICOM geometry when available.
4. Validate that geometry against the physical CT lung.
5. If DICOM geometry has a translation problem, refine it.
6. If DICOM geometry remains inconsistent, perform a registration
   rescue using in-plane rotations/flips/axis swaps and CT-lung
   centre initialisation.
7. Resample the registered SPECT onto the CT grid.
8. Load the DICOM SEGMENT treatment volume.
9. Distribute the full 1 GBq uniformly within SEGMENT.
10. Density-scale the Graves DPK.
11. Convolve activity with the DPK.
12. Calculate dose from the lung voxel mass.

MCDB S-values are NOT used.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import gc
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from scipy import ndimage
from scipy.signal import oaconvolve
from skimage.measure import marching_cubes

import matplotlib.pyplot as plt


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)


# =============================================================================
# SETTINGS
# =============================================================================

DICOM_ROOT = Path(
    "/Users/yaser/Documents/Python_scripts/Dale/"
    "LUNG-ABLATION-TEST-SET-1/"
    "1.2.826.0.1.5968184.2.2.1.1750921566855.188412"
)

KERNEL_DIR = Path(
    "/Users/yaser/Documents/Python_scripts/Playground/"
    "20190213_Graves_DosePointKernels_v1.0"
)

KERNEL_FILES = {
    "Y90": KERNEL_DIR / "90Y_64.10H_beta.io_processed.csv",
    "Lu177": KERNEL_DIR / "177LU_6.73D_beta.io_processed.csv",
}

OUTPUT_DIR = (
    DICOM_ROOT.parent /
    "LUNG_ABLATION_DOSIMETRY_RESULTS"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INITIAL_ACTIVITY_GBq = 1.0

DECAY_FRACTION = 0.98

HALF_LIFE_DAYS = {
    "Y90": 2.6684,
    "Lu177": 6.647,
}

LUNG_HU_MIN = -1000
LUNG_HU_MAX = -250

MIN_LUNG_COMPONENT_VOXELS = 1000

WATER_DENSITY_G_CM3 = 1.00
LUNG_DENSITY_G_CM3 = 0.26

# CT-HU-derived density limits used for voxel-wise dose conversion.
# These are taken directly from the previous CT-density implementation.
DENSITY_MIN_G_CM3 = 0.05
DENSITY_MAX_G_CM3 = 1.20

# Reference treated SEGMENT geometry for this dataset. These values are used
# for QA/reporting only; the voxelised CT lung mask remains the quantitative
# geometry used by the dose calculation.
EXPECTED_SEGMENT_VOLUME_CM3 = 181.15
EXPECTED_SEGMENT_MASS_G = (
    EXPECTED_SEGMENT_VOLUME_CM3 * LUNG_DENSITY_G_CM3
)
EXPECTED_SEGMENT_VOLUME_TOLERANCE_FRACTION = 0.05

# Approximate anatomical total lung volume supplied for this dataset:
# treated segment (~181.15 cc) + left lung excluding segment (~945 cc)
# + right lung (~1350 cc). Used for whole-lung QA only.
EXPECTED_RIGHT_LUNG_VOLUME_CM3 = 1681.5
EXPECTED_LEFT_LUNG_VOLUME_CM3 = 1375.9
EXPECTED_TOTAL_LUNG_VOLUME_CM3 = (
    EXPECTED_RIGHT_LUNG_VOLUME_CM3
    + EXPECTED_LEFT_LUNG_VOLUME_CM3
)
EXPECTED_TOTAL_LUNG_MASS_G = (
    EXPECTED_TOTAL_LUNG_VOLUME_CM3 * LUNG_DENSITY_G_CM3
)
EXPECTED_TOTAL_LUNG_VOLUME_TOLERANCE_FRACTION = 0.002

# The same fixed anatomical geometry is used for BOTH Y-90 and Lu-177.
# This correction is performed once, before either isotope is calculated.
APPLY_REFERENCE_LUNG_VOLUME_CORRECTION = True

# SEGMENT geometry is expected to be preserved through registration/resampling.
# Do not artificially dilate or shrink the SEGMENT. A large loss indicates a
# registration/FOV problem and should stop the analysis.
FAIL_ON_SEGMENT_VOLUME_LOSS = True

# For the anatomical lung mask, the 2476.15 cc value is an external QA
# reference rather than a forced target. Keep the tolerance relatively broad
# because CT segmentation thresholds and inclusion/exclusion of airways can
# change the measured anatomical volume.
FAIL_ON_LUNG_VOLUME_QA = True

DENSITY_SCALE_KERNEL = True

TARGET_MASK_NPY = None

# The DICOM SEGMENT is the treatment/activity volume.
USE_DICOM_SEGMENT = True

# Default: all 1 GBq is uniformly distributed within SEGMENT.
# Optional alternative: "SPECT_WEIGHTED_SEGMENT" keeps the SEGMENT
# as the hard activity boundary but weights activity by registered MAA SPECT.
ACTIVITY_DISTRIBUTION_MODE = "UNIFORM_SEGMENT"

# Absolute dose levels for EBRT-style isodose contours.
DOSE_CONTOUR_LEVELS_GY = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
)

# Dose-volume levels reported for whole lung, treated SEGMENT,
# and non-target lung.
DOSE_VOLUME_LEVELS_GY = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
)

SPECT_POSITIVE_ONLY = True

KERNEL_ENERGY_FRACTION = 0.999
KERNEL_MAX_RADIUS_CM = None

CROP_TO_LUNG = True
EXTRA_CROP_MARGIN_VOXELS = 2

FLOAT_DTYPE = np.float32

AUTO_REGISTER_SPECT = True

# -------------------------------------------------------------------------
# Registration settings
# -------------------------------------------------------------------------

REGISTRATION_COARSE_STEP_MM = 20.0
REGISTRATION_FINE_STEP_MM = 5.0
REGISTRATION_FINAL_STEP_MM = 1.0

REGISTRATION_COARSE_RANGE_MM = 100.0
REGISTRATION_FINE_RANGE_MM = 20.0
REGISTRATION_FINAL_RANGE_MM = 5.0

REGISTRATION_SUBSAMPLE = 2

MIN_REGISTRATION_OVERLAP_FRACTION = 0.50

# If DICOM geometry does not place at least this fraction of positive
# SPECT activity within the CT lung, registration rescue is triggered.
DICOM_GEOMETRY_RESCUE = True

# Number of best orientation candidates retained after the initial
# screening before translation optimisation.
ORIENTATION_REFINEMENT_CANDIDATES = 4

# -------------------------------------------------------------------------
# Figure settings
# -------------------------------------------------------------------------

FIG_DPI = 600

DOSE_VMIN = 0.0
DOSE_VMAX = 50

# Maximum voxel count used for 3-D visualisation resampling.
MAX_3D_VIS_VOXELS = 2_000_000

# 3-D isodose surfaces to display. Only levels below the actual dose
# maximum are rendered.
DOSE_3D_SURFACE_LEVELS_GY = (
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
)


# =============================================================================
# CONSTANTS
# =============================================================================

MEV_TO_J = 1.602176634e-13

MEV_PER_G_TO_GY = 1.602176634e-10

LN2 = np.log(2.0)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def print_header(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_memory(name, array):

    if array is None:
        return

    if not isinstance(array, np.ndarray):
        return

    memory_mb = array.nbytes / (1024 ** 2)

    print(
        f"{name:<40s} "
        f"shape={str(array.shape):<22s} "
        f"dtype={str(array.dtype):<10s} "
        f"memory={memory_mb:,.1f} MB"
    )


def require_file(path, description):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )

    if not path.is_file():

        raise FileNotFoundError(
            f"{description} is not a file:\n{path}"
        )


def normalise_vector(vector):

    vector = np.asarray(
        vector,
        dtype=np.float64
    )

    norm = np.linalg.norm(
        vector
    )

    if norm < 1e-12:

        raise RuntimeError(
            "Cannot normalise zero-length vector."
        )

    return vector / norm


# =============================================================================
# DICOM BASIC GEOMETRY
# =============================================================================

def get_float(
    ds,
    keyword,
    default=None
):

    value = getattr(
        ds,
        keyword,
        None
    )

    if value is None:
        return default

    try:
        return float(value)

    except Exception:
        return default


def get_vector(
    ds,
    keyword
):

    value = getattr(
        ds,
        keyword,
        None
    )

    if value is None:
        return None

    try:

        return np.asarray(
            [
                float(x)
                for x in value
            ],
            dtype=np.float64
        )

    except Exception:

        return None


def _as_float_vector(
    value,
    length
):

    if value is None:
        return None

    try:

        arr = np.asarray(
            [
                float(x)
                for x in value
            ],
            dtype=np.float64
        )

    except Exception:

        return None

    if arr.size != length:
        return None

    if not np.all(
        np.isfinite(arr)
    ):
        return None

    return arr


# =============================================================================
# DICOM DISCOVERY
# =============================================================================

def discover_dicom_series(root):

    print_header(
        "DICOM SERIES DISCOVERY"
    )

    root = Path(root)

    if not root.exists():

        raise FileNotFoundError(
            f"DICOM root does not exist:\n{root}"
        )

    if not root.is_dir():

        raise NotADirectoryError(
            f"DICOM root is not a directory:\n{root}"
        )

    series = {}

    all_files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True
            )

        except Exception:

            continue

        uid = getattr(
            ds,
            "SeriesInstanceUID",
            None
        )

        if uid is None:
            continue

        series.setdefault(
            uid,
            []
        ).append(
            path
        )

        all_files.append(
            path
        )

    print(
        f"Discovered DICOM files: {len(all_files):,}"
    )

    print(
        f"Discovered series: {len(series):,}"
    )

    series_list = []

    for uid, files in series.items():

        first_ds = None

        for path in files:

            try:

                first_ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=True
                )

                break

            except Exception:

                continue

        if first_ds is None:
            continue

        modality = str(
            getattr(
                first_ds,
                "Modality",
                ""
            )
        )

        description = str(
            getattr(
                first_ds,
                "SeriesDescription",
                ""
            )
        )

        series_list.append(
            {
                "uid": uid,
                "files": files,
                "modality": modality,
                "description": description,
            }
        )

        print()
        print(
            f"Modality    : {modality}"
        )

        print(
            f"Description : {description}"
        )

        print(
            f"UID         : {uid}"
        )

        print(
            f"Files       : {len(files):,}"
        )

    return series_list


# =============================================================================
# CT SERIES SELECTION
# =============================================================================

def select_ct_series(
    series_list
):

    candidates = [
        s
        for s in series_list
        if s["modality"].upper() == "CT"
    ]

    if not candidates:

        raise RuntimeError(
            "No CT DICOM series found."
        )

    preferred = []

    for s in candidates:

        description = (
            s["description"]
            .lower()
        )

        if (
            "rns_spect_ac_lungs" in description
            or
            "ct expiration" in description
        ):

            preferred.append(
                s
            )

    if preferred:

        selected = max(
            preferred,
            key=lambda x: len(x["files"])
        )

    else:

        selected = max(
            candidates,
            key=lambda x: len(x["files"])
        )

    print()
    print(
        "Selected CT series:"
    )

    print(
        f"    {selected['description']}"
    )

    print(
        f"    {selected['uid']}"
    )

    print(
        f"    {len(selected['files']):,} files"
    )

    return selected


# =============================================================================
# LOAD CT SERIES
# =============================================================================

def load_ct_series(
    series
):

    print_header(
        "LOADING CT"
    )

    datasets = []

    for path in series["files"]:

        try:

            ds = pydicom.dcmread(
                str(path),
                force=True
            )

        except Exception:

            continue

        if not hasattr(
            ds,
            "PixelData"
        ):
            continue

        datasets.append(
            ds
        )

    if not datasets:

        raise RuntimeError(
            "No CT images with PixelData were found."
        )

    first = datasets[0]

    iop = _as_float_vector(
        getattr(
            first,
            "ImageOrientationPatient",
            None
        ),
        6
    )

    if iop is None:

        raise RuntimeError(
            "CT ImageOrientationPatient is missing."
        )

    row_cos = normalise_vector(
        iop[:3]
    )

    col_cos = normalise_vector(
        iop[3:]
    )

    normal = normalise_vector(
        np.cross(
            row_cos,
            col_cos
        )
    )

    pixel_spacing = _as_float_vector(
        getattr(
            first,
            "PixelSpacing",
            None
        ),
        2
    )

    if pixel_spacing is None:

        raise RuntimeError(
            "CT PixelSpacing is missing."
        )

    row_spacing = float(
        pixel_spacing[0]
    )

    col_spacing = float(
        pixel_spacing[1]
    )

    slice_information = []

    for ds in datasets:

        ipp = _as_float_vector(
            getattr(
                ds,
                "ImagePositionPatient",
                None
            ),
            3
        )

        if ipp is None:
            continue

        projection = float(
            np.dot(
                ipp,
                normal
            )
        )

        instance = int(
            getattr(
                ds,
                "InstanceNumber",
                0
            )
        )

        slice_information.append(
            (
                projection,
                instance,
                ipp,
                ds
            )
        )

    if not slice_information:

        raise RuntimeError(
            "CT ImagePositionPatient is missing."
        )

    slice_information.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    positions = [
        item[2]
        for item in slice_information
    ]

    datasets_sorted = [
        item[3]
        for item in slice_information
    ]

    if len(positions) > 1:

        projections = np.asarray(
            [
                np.dot(
                    p,
                    normal
                )
                for p in positions
            ],
            dtype=np.float64
        )

        differences = np.diff(
            projections
        )

        differences = np.abs(
            differences
        )

        differences = differences[
            differences > 1e-6
        ]

        if differences.size:

            slice_spacing = float(
                np.median(
                    differences
                )
            )

        else:

            slice_spacing = get_float(
                first,
                "SliceThickness",
                1.0
            )

    else:

        slice_spacing = get_float(
            first,
            "SliceThickness",
            1.0
        )

    arrays = []

    for ds in datasets_sorted:

        pixel = ds.pixel_array.astype(
            np.float32
        )

        slope = get_float(
            ds,
            "RescaleSlope",
            1.0
        )

        intercept = get_float(
            ds,
            "RescaleIntercept",
            0.0
        )

        pixel = (
            pixel * slope
            + intercept
        )

        arrays.append(
            pixel
        )

    ct_hu = np.stack(
        arrays,
        axis=0
    ).astype(
        FLOAT_DTYPE
    )

    origin = np.asarray(
        positions[0],
        dtype=np.float64
    )

    affine_xyz = np.column_stack(
        [
            col_cos * col_spacing,
            row_cos * row_spacing,
            normal * slice_spacing,
        ]
    )

    frame_uid = getattr(
        first,
        "FrameOfReferenceUID",
        None
    )

    print(
        f"CT shape                 : {ct_hu.shape}"
    )

    print(
        f"CT voxel spacing (mm)    : "
        f"{slice_spacing:.6f}, "
        f"{row_spacing:.6f}, "
        f"{col_spacing:.6f}"
    )

    print(
        f"CT origin (mm)           : {origin}"
    )

    print(
        f"CT row direction         : {row_cos}"
    )

    print(
        f"CT column direction      : {col_cos}"
    )

    print(
        f"CT slice direction       : {normal}"
    )

    print(
        f"FrameOfReferenceUID      : {frame_uid}"
    )

    print_memory(
        "CT HU",
        ct_hu
    )

    return (
        ct_hu,
        affine_xyz,
        origin,
        col_spacing,
        row_spacing,
        slice_spacing,
        row_cos,
        col_cos,
        normal,
        frame_uid,
    )


# =============================================================================
# LUNG MASK
# =============================================================================

def make_lung_mask(
    ct_hu
):
    """
    Robust anatomical lung-mask extraction from CT.

    The previous implementation restricted the -1000 to -250 HU candidate
    to a rectangular body bounding box. That allowed external/background air
    inside the box to be classified as lung and produced an unphysical
    ~26,000 cc lung volume.

    This implementation instead:
        1. Builds a 3-D patient-body mask from HU > -500.
        2. Keeps the largest connected body component.
        3. Fills the body volume so the internal thoracic air spaces are
           available as candidate lung.
        4. Intersects the -1000 to -250 HU candidate with the filled body.
        5. Separates the left and right lungs using the body-centre x plane
           and keeps the dominant connected lung component on each side.
        6. Applies only light morphology.

    This deliberately does NOT force the lung volume to the expected
    2476.15 cc. The expected value is used only for QA.
    """

    print_header("LUNG SEGMENTATION — ANATOMICAL BODY-CONSTRAINED")

    # ------------------------------------------------------------------
    # 1. Low-HU candidate: aerated lung / airway air
    # ------------------------------------------------------------------
    candidate = (
        np.isfinite(ct_hu)
        &
        (ct_hu >= LUNG_HU_MIN)
        &
        (ct_hu <= LUNG_HU_MAX)
    )

    # ------------------------------------------------------------------
    # 2. Patient body mask
    # ------------------------------------------------------------------
    body_candidate = (
        np.isfinite(ct_hu)
        &
        (ct_hu > -500.0)
    )

    # A small closing bridges tiny discontinuities in the external body
    # contour without substantially changing the anatomy.
    body_structure = ndimage.generate_binary_structure(3, 1)
    body_candidate = ndimage.binary_closing(
        body_candidate,
        structure=body_structure,
        iterations=2
    )

    body_labels, body_n = ndimage.label(
        body_candidate,
        structure=body_structure
    )

    if body_n == 0:
        raise RuntimeError("Could not identify patient body.")

    body_sizes = np.bincount(body_labels.ravel())
    body_sizes[0] = 0
    largest_body = int(np.argmax(body_sizes))

    body_mask = body_labels == largest_body

    if not np.any(body_mask):
        raise RuntimeError("Largest body component is empty.")

    # Fill internal cavities. This prevents external air from entering the
    # lung candidate while allowing the aerated lung regions inside the body
    # to be selected.
    body_filled = ndimage.binary_fill_holes(body_mask)

    # Restrict candidate to the patient interior.
    thoracic_air = candidate & body_filled

    # ------------------------------------------------------------------
    # 3. Light morphology on the candidate
    # ------------------------------------------------------------------
    thoracic_air = ndimage.binary_closing(
        thoracic_air,
        structure=body_structure,
        iterations=1
    )

    thoracic_air = ndimage.binary_opening(
        thoracic_air,
        structure=body_structure,
        iterations=1
    )

    # ------------------------------------------------------------------
    # 4. Determine the body centre in X and retain dominant component on
    #    each side. This prevents a large connected airway/background
    #    component from becoming the whole-lung mask.
    #
    #    NumPy axis 2 is X.
    # ------------------------------------------------------------------
    body_indices = np.argwhere(body_mask)

    x_min = int(body_indices[:, 2].min())
    x_max = int(body_indices[:, 2].max())
    x_mid = 0.5 * (x_min + x_max)

    left_candidate = thoracic_air.copy()
    right_candidate = thoracic_air.copy()

    left_candidate[:, :, int(np.ceil(x_mid)):] = False
    right_candidate[:, :, :int(np.floor(x_mid)) + 1] = False

    def largest_component(mask, minimum_voxels):
        labels, n = ndimage.label(
            mask,
            structure=body_structure
        )

        if n == 0:
            return np.zeros_like(mask, dtype=bool)

        sizes = np.bincount(labels.ravel())
        sizes[0] = 0

        label_id = int(np.argmax(sizes))

        if sizes[label_id] < minimum_voxels:
            return np.zeros_like(mask, dtype=bool)

        return labels == label_id

    left_lung = largest_component(
        left_candidate,
        MIN_LUNG_COMPONENT_VOXELS
    )

    right_lung = largest_component(
        right_candidate,
        MIN_LUNG_COMPONENT_VOXELS
    )

    # If one side unexpectedly fails, fall back to the largest components
    # of the complete body-constrained candidate rather than silently
    # producing a partial lung.
    if not np.any(left_lung) or not np.any(right_lung):

        labels, n = ndimage.label(
            thoracic_air,
            structure=body_structure
        )

        if n == 0:
            raise RuntimeError(
                "Could not identify any body-constrained lung components."
            )

        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]

        selected = [
            int(i)
            for i in order
            if sizes[i] >= MIN_LUNG_COMPONENT_VOXELS
        ][:2]

        if len(selected) < 2:
            raise RuntimeError(
                "Could not identify two anatomical lung components."
            )

        lung_mask = np.isin(labels, selected)

    else:
        lung_mask = left_lung | right_lung

    # Remove isolated tiny fragments after component selection.
    labels_final, n_final = ndimage.label(
        lung_mask,
        structure=body_structure
    )

    if n_final > 0:
        sizes_final = np.bincount(labels_final.ravel())
        sizes_final[0] = 0

        keep = np.where(
            sizes_final >= MIN_LUNG_COMPONENT_VOXELS
        )[0]

        lung_mask = np.isin(
            labels_final,
            keep
        )

    # ------------------------------------------------------------------
    # 5. Report QA
    # ------------------------------------------------------------------
    values = ct_hu[lung_mask]

    print(f"Body voxels              : {np.count_nonzero(body_mask):,}")
    print(f"Body-filled voxels       : {np.count_nonzero(body_filled):,}")
    print(f"Body-constrained air     : {np.count_nonzero(thoracic_air):,}")
    print(f"Left lung voxels         : {np.count_nonzero(left_lung):,}")
    print(f"Right lung voxels        : {np.count_nonzero(right_lung):,}")
    print(f"Final lung voxels        : {np.count_nonzero(lung_mask):,}")

    if values.size:
        print(
            f"Lung HU range            : "
            f"{np.min(values):.1f} to {np.max(values):.1f}"
        )
        print(
            f"Lung HU median           : "
            f"{np.median(values):.1f}"
        )

    return lung_mask


# =============================================================================
# =============================================================================
# FIXED REFERENCE LUNG GEOMETRY CORRECTION / AUDIT
# =============================================================================

def correct_lung_mask_to_reference_geometry(
    ct_hu,
    lung_mask,
    ct_affine,
    ct_voxel_volume_cm3,
):
    """Make one fixed, isotope-independent anatomical lung mask.

    The existing anatomical lung segmentation is preserved as the starting
    point.  Laterality is determined from the two largest connected
    components of that existing lung mask using physical DICOM patient-X
    coordinates.  The supplied reference volumes are then used only to
    correct
    the volume of each anatomical lung separately.

    This avoids re-splitting the lung mask at the centre of the whole body,
    which can incorrectly divide one lung into a ~900 cc / ~2022 cc pair.
    """

    if not APPLY_REFERENCE_LUNG_VOLUME_CORRECTION:
        return lung_mask.astype(bool), None, None

    if ct_voxel_volume_cm3 <= 0:
        raise RuntimeError("Invalid CT voxel volume.")

    target_right = int(np.rint(
        EXPECTED_RIGHT_LUNG_VOLUME_CM3 / ct_voxel_volume_cm3
    ))
    target_left = int(np.rint(
        EXPECTED_LEFT_LUNG_VOLUME_CM3 / ct_voxel_volume_cm3
    ))

    structure = ndimage.generate_binary_structure(3, 1)

    # ------------------------------------------------------------------
    # Build the same body-constrained low-HU candidate used by the existing
    # lung segmentation.  This candidate is used only for adding voxels
    # around the already identified anatomical lung components.
    # ------------------------------------------------------------------
    body_candidate = (
        np.isfinite(ct_hu)
        & (ct_hu > -500.0)
    )

    body_candidate = ndimage.binary_closing(
        body_candidate,
        structure=structure,
        iterations=2,
    )

    labels, n = ndimage.label(
        body_candidate,
        structure=structure,
    )

    if n == 0:
        raise RuntimeError(
            "Could not identify patient body for lung-volume correction."
        )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body_mask = labels == int(np.argmax(sizes))
    body_filled = ndimage.binary_fill_holes(body_mask)

    candidate = (
        np.isfinite(ct_hu)
        & (ct_hu >= LUNG_HU_MIN)
        & (ct_hu <= LUNG_HU_MAX)
        & body_filled
    )

    # ------------------------------------------------------------------
    # Identify the two anatomical lungs from the EXISTING lung mask.
    #
    # Do not divide the whole body at its X midpoint.  The existing
    # segmentation already contains the anatomical lung components.
    # ------------------------------------------------------------------
    existing_labels, existing_n = ndimage.label(
        lung_mask.astype(bool),
        structure=structure,
    )

    if existing_n < 2:
        raise RuntimeError(
            "The existing lung mask does not contain two connected "
            "anatomical lung components. "
            f"Found {existing_n} component(s)."
        )

    existing_sizes = np.bincount(existing_labels.ravel())
    existing_sizes[0] = 0

    component_order = np.argsort(existing_sizes)[::-1]
    component_ids = [
        int(label_id)
        for label_id in component_order
        if existing_sizes[label_id] > 0
    ][:2]

    if len(component_ids) != 2:
        raise RuntimeError(
            "Could not identify two anatomical lung components from the "
            "existing lung mask."
        )

    component_a = existing_labels == component_ids[0]
    component_b = existing_labels == component_ids[1]

    # ------------------------------------------------------------------
    # Calculate physical DICOM patient-X centroid for each existing lung.
    #
    # In DICOM patient coordinates, smaller X corresponds to patient RIGHT
    # and larger X corresponds to patient LEFT for this CT orientation.
    # ------------------------------------------------------------------
    def physical_x_for_mask(mask):
        idx = np.argwhere(mask)
        if idx.size == 0:
            raise RuntimeError("Empty lung component encountered.")

        x = (
            float(ct_affine[0, 0]) * idx[:, 2]
            + float(ct_affine[0, 1]) * idx[:, 1]
            + float(ct_affine[0, 2]) * idx[:, 0]
        )
        return float(np.mean(x))

    component_a_x = physical_x_for_mask(component_a)
    component_b_x = physical_x_for_mask(component_b)

    if component_a_x <= component_b_x:
        right_current = component_a
        left_current = component_b
        right_x = component_a_x
        left_x = component_b_x
        right_component_label = component_ids[0]
        left_component_label = component_ids[1]
    else:
        right_current = component_b
        left_current = component_a
        right_x = component_b_x
        left_x = component_a_x
        right_component_label = component_ids[1]
        left_component_label = component_ids[0]

    # ------------------------------------------------------------------
    # Build an anatomical separation plane halfway between the actual
    # right- and left-lung centroids.  This is only used to constrain added
    # candidate voxels; it is NOT used to redefine the existing lungs.
    # ------------------------------------------------------------------
    x_mid_lungs = 0.5 * (right_x + left_x)

    zz, yy, xx = np.indices(
        ct_hu.shape,
        dtype=np.float32,
    )

    patient_x = (
        float(ct_affine[0, 0]) * xx
        + float(ct_affine[0, 1]) * yy
        + float(ct_affine[0, 2]) * zz
    )

    right_selector = patient_x <= x_mid_lungs
    left_selector = patient_x > x_mid_lungs

    right_candidate = candidate & right_selector
    left_candidate = candidate & left_selector

    right_current_volume = (
        np.count_nonzero(right_current)
        * ct_voxel_volume_cm3
    )
    left_current_volume = (
        np.count_nonzero(left_current)
        * ct_voxel_volume_cm3
    )

    print()
    print("LUNG SIDE ASSIGNMENT AUDIT")
    print(
        f"    Existing RIGHT component : "
        f"{np.count_nonzero(right_current):,} voxels; "
        f"{right_current_volume:.6f} cc"
    )
    print(
        f"    Existing LEFT component  : "
        f"{np.count_nonzero(left_current):,} voxels; "
        f"{left_current_volume:.6f} cc"
    )
    print(
        f"    RIGHT physical-X centroid: "
        f"{right_x:.6f} mm"
    )
    print(
        f"    LEFT physical-X centroid : "
        f"{left_x:.6f} mm"
    )
    print(
        f"    Lung-centroid X midpoint : "
        f"{x_mid_lungs:.6f} mm"
    )
    print(
        f"    RIGHT component label    : "
        f"{right_component_label}"
    )
    print(
        f"    LEFT component label     : "
        f"{left_component_label}"
    )

    def correct_side(current, candidate_side, target, name):
        current = current.astype(bool)
        current_idx = np.argwhere(current)
        current_n = current_idx.shape[0]

        if current_n == target:
            return current.copy()

        if current_n > target:
            # Remove the most peripheral voxels first by retaining the
            # central part of the existing anatomical component.
            centre = np.mean(
                current_idx.astype(np.float64),
                axis=0,
            )

            dist2 = np.sum(
                (
                    current_idx.astype(np.float64)
                    - centre
                ) ** 2,
                axis=1,
            )

            selected = current_idx[
                np.argsort(dist2)[:target]
            ]

            corrected = np.zeros_like(
                current,
                dtype=bool,
            )

            corrected[
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
            ] = True

            return corrected

        need = target - current_n

        addition = candidate_side & (~current)
        available = int(
            np.count_nonzero(addition)
        )

        if available < need:
            raise RuntimeError(
                f"{name}: cannot reach reference volume. "
                f"Need {need:,} additional voxels but only "
                f"{available:,} anatomically constrained candidates "
                f"are available. "
                f"Current volume = "
                f"{current_n * ct_voxel_volume_cm3:.6f} cc; "
                f"target = "
                f"{target * ct_voxel_volume_cm3:.6f} cc."
            )

        distance = ndimage.distance_transform_edt(
            ~current
        )

        idx = np.argwhere(addition)

        d = distance[addition]

        hu = ct_hu[addition].astype(
            np.float64
        )

        # Nearest to the existing lung first; lower HU breaks ties.
        order = np.lexsort(
            (
                hu,
                d,
            )
        )

        selected = idx[
            order[:need]
        ]

        corrected = current.copy()

        corrected[
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
        ] = True

        return corrected

    # ------------------------------------------------------------------
    # Correct each anatomical lung independently.
    # ------------------------------------------------------------------
    right = correct_side(
        right_current,
        right_candidate,
        target_right,
        "RIGHT LUNG",
    )

    left = correct_side(
        left_current,
        left_candidate,
        target_left,
        "LEFT LUNG",
    )

    corrected = right | left

    measured = {
        "Right lung": (
            np.count_nonzero(right)
            * ct_voxel_volume_cm3
        ),
        "Left lung": (
            np.count_nonzero(left)
            * ct_voxel_volume_cm3
        ),
        "Whole lung": (
            np.count_nonzero(corrected)
            * ct_voxel_volume_cm3
        ),
    }

    expected = {
        "Right lung": EXPECTED_RIGHT_LUNG_VOLUME_CM3,
        "Left lung": EXPECTED_LEFT_LUNG_VOLUME_CM3,
        "Whole lung": EXPECTED_TOTAL_LUNG_VOLUME_CM3,
    }

    print()
    print("FIXED REFERENCE LUNG GEOMETRY")

    for name in (
        "Right lung",
        "Left lung",
        "Whole lung",
    ):
        diff = (
            100.0
            * (
                measured[name]
                / expected[name]
                - 1.0
            )
        )

        print(
            f"    {name:12s}: "
            f"{measured[name]:.6f} cc "
            f"(reference "
            f"{expected[name]:.6f} cc; "
            f"{diff:+.6f}%)"
        )

        if (
            abs(diff)
            >
            100.0
            * EXPECTED_TOTAL_LUNG_VOLUME_TOLERANCE_FRACTION
        ):
            raise RuntimeError(
                f"{name} volume failed the fixed reference "
                "geometry check."
            )

    if np.any(right & left):
        raise RuntimeError(
            "Corrected RIGHT and LEFT lung masks overlap."
        )

    if np.count_nonzero(corrected) != (
        np.count_nonzero(right)
        + np.count_nonzero(left)
    ):
        raise RuntimeError(
            "Whole-lung voxel closure failed."
        )

    return (
        corrected.astype(bool),
        right.astype(bool),
        left.astype(bool),
    )

# =============================================================================
# LUNG PHYSICAL CENTRE
# =============================================================================

def calculate_lung_physical_centre(
    lung_mask,
    ct_affine,
    ct_origin
):

    indices = np.argwhere(
        lung_mask
    )

    if indices.size == 0:

        raise RuntimeError(
            "Cannot calculate lung centre from empty mask."
        )

    centre_zyx = np.mean(
        indices,
        axis=0
    )

    centre_xyz_index = np.array(
        [
            centre_zyx[2],
            centre_zyx[1],
            centre_zyx[0],
        ],
        dtype=np.float64
    )

    centre_patient = (
        ct_origin
        +
        ct_affine @ centre_xyz_index
    )

    return centre_patient


# =============================================================================
# TARGET MASK
# =============================================================================

def load_target_mask(
    target_path,
    ct_shape
):

    if target_path is None:
        return None

    target_path = Path(
        target_path
    )

    require_file(
        target_path,
        "Target mask"
    )

    mask = np.load(
        target_path
    )

    if mask.shape != tuple(
        ct_shape
    ):

        raise RuntimeError(
            "Target mask shape does not match CT.\n"
            f"Target: {mask.shape}\n"
            f"CT: {ct_shape}"
        )

    mask = (
        mask > 0
    )

    print()
    print(
        f"Target mask voxels: "
        f"{np.count_nonzero(mask):,}"
    )

    return mask


# =============================================================================
# SPECT SERIES SELECTION
# =============================================================================

def select_spect_series(
    series_list
):

    candidates = [
        s
        for s in series_list
        if s["modality"].upper() == "NM"
    ]

    if not candidates:

        raise RuntimeError(
            "No NM/SPECT DICOM series found."
        )

    preferred = []

    for s in candidates:

        description = (
            s["description"]
            .lower()
        )

        if (
            "perf tomo" in description
            and
            "ac" in description
        ):

            preferred.append(
                s
            )

    if preferred:

        selected = max(
            preferred,
            key=lambda x: len(x["files"])
        )

    else:

        non_segment = [
            s
            for s in candidates
            if "segment" not in
            s["description"].lower()
        ]

        if non_segment:

            selected = max(
                non_segment,
                key=lambda x: len(x["files"])
            )

        else:

            selected = max(
                candidates,
                key=lambda x: len(x["files"])
            )

    print()
    print(
        "Selected SPECT series:"
    )

    print(
        f"    {selected['description']}"
    )

    print(
        f"    {selected['uid']}"
    )

    print(
        f"    {len(selected['files']):,} files"
    )

    return selected


# =============================================================================
# DICOM SEGMENT SERIES SELECTION
# =============================================================================

def select_segment_series(series_list):
    """
    Select the NM DICOM series whose SeriesDescription identifies it as
    SEGMENT. This is the treatment/activity volume and is independent of
    the reconstructed MAA SPECT used for registration/QC.
    """

    candidates = [
        s
        for s in series_list
        if s["modality"].upper() == "NM"
        and "segment" in s["description"].lower()
    ]

    if not candidates:
        raise RuntimeError(
            "No NM DICOM SEGMENT series was found. "
            "The treatment volume is required for this dosimetry workflow."
        )

    selected = max(
        candidates,
        key=lambda x: len(x["files"])
    )

    print()
    print("Selected DICOM SEGMENT series:")
    print(f"    {selected['description']}")
    print(f"    {selected['uid']}")
    print(f"    {len(selected['files']):,} file(s)")

    return selected


# =============================================================================
# DICOM SEGMENT -> SPECT/CT GEOMETRY
# =============================================================================

def _sitk_geometry_to_affine(image):
    """
    Convert SimpleITK image geometry to the affine convention used by this
    script: patient_xyz = origin_xyz + affine_xyz @ index_xyz.
    """

    spacing_xyz = np.asarray(
        image.GetSpacing(),
        dtype=np.float64
    )

    direction = np.asarray(
        image.GetDirection(),
        dtype=np.float64
    ).reshape(3, 3)

    affine_xyz = (
        direction
        @
        np.diag(spacing_xyz)
    )

    origin_xyz = np.asarray(
        image.GetOrigin(),
        dtype=np.float64
    )

    return (
        affine_xyz,
        origin_xyz,
        spacing_xyz,
        direction
    )


def load_and_align_dicom_segment(
    segment_series,
    spect_series
):
    """
    Load the DICOM SEGMENT volume and reconcile it with the reconstructed
    Perf Tomo grid.

    For the supplied dataset, SEGMENT and Perf Tomo have the same matrix
    and in-plane geometry but opposite frame ordering in Z. The physical
    Z-span is approximately (N-1)*spacing, so SEGMENT is reversed along
    the NumPy Z axis and then assigned the Perf Tomo geometry.

    The quantitative SEGMENT values are never smoothed.
    """

    print_header(
        "LOADING DICOM SEGMENT"
    )

    if len(segment_series["files"]) == 0:
        raise RuntimeError(
            "SEGMENT series contains no files."
        )

    segment_path = sorted(
        segment_series["files"],
        key=lambda x: str(x)
    )[0]

    spect_path = sorted(
        spect_series["files"],
        key=lambda x: str(x)
    )[0]

    segment_image = sitk.ReadImage(
        str(segment_path)
    )

    spect_image = sitk.ReadImage(
        str(spect_path)
    )

    segment_array = sitk.GetArrayFromImage(
        segment_image
    ).astype(
        np.float32
    )

    spect_array = sitk.GetArrayFromImage(
        spect_image
    )

    if segment_array.ndim != 3:
        raise RuntimeError(
            f"Expected 3-D SEGMENT image, got {segment_array.shape}."
        )

    if segment_array.shape != spect_array.shape:
        raise RuntimeError(
            "SEGMENT and Perf Tomo matrix sizes differ.\n"
            f"SEGMENT: {segment_array.shape}\n"
            f"Perf Tomo: {spect_array.shape}\n"
            "The automatic frame-order correction is therefore unsafe."
        )

    seg_spacing = np.asarray(
        segment_image.GetSpacing(),
        dtype=np.float64
    )

    perf_spacing = np.asarray(
        spect_image.GetSpacing(),
        dtype=np.float64
    )

    seg_origin = np.asarray(
        segment_image.GetOrigin(),
        dtype=np.float64
    )

    perf_origin = np.asarray(
        spect_image.GetOrigin(),
        dtype=np.float64
    )

    seg_direction = np.asarray(
        segment_image.GetDirection(),
        dtype=np.float64
    ).reshape(3, 3)

    perf_direction = np.asarray(
        spect_image.GetDirection(),
        dtype=np.float64
    ).reshape(3, 3)

    print(
        f"SEGMENT array shape : {segment_array.shape}"
    )
    print(
        f"SEGMENT spacing     : {seg_spacing}"
    )
    print(
        f"SEGMENT origin      : {seg_origin}"
    )
    print(
        f"Perf Tomo origin    : {perf_origin}"
    )

    z_spacing = float(
        0.5 * (
            abs(seg_spacing[2])
            +
            abs(perf_spacing[2])
        )
    )

    z_origin_difference = float(
        np.linalg.norm(
            seg_origin
            -
            perf_origin
        )
    )

    expected_z_span = (
        (segment_array.shape[0] - 1)
        *
        z_spacing
    )

    print(
        f"SEGMENT/Perf origin separation: "
        f"{z_origin_difference:.6f} mm"
    )
    print(
        f"Expected one-volume Z span    : "
        f"{expected_z_span:.6f} mm"
    )

    same_spacing = np.allclose(
        seg_spacing,
        perf_spacing,
        atol=0.01,
        rtol=0.0
    )

    same_xy_direction = np.allclose(
        seg_direction[:, :2],
        perf_direction[:, :2],
        atol=2e-3,
        rtol=0.0
    )

    # In this dataset the SEGMENT image is the same physical reconstruction
    # grid as Perf Tomo, but its frame direction is reversed.
    reverse_z = (
        same_spacing
        and
        same_xy_direction
        and
        abs(
            z_origin_difference
            -
            expected_z_span
        )
        < max(
            1.0,
            0.02 * expected_z_span
        )
    )

    if reverse_z:
        print(
            "SEGMENT geometry check: reverse-Z frame ordering detected."
        )
        corrected_array = segment_array[
            ::-1,
            :,
            :
        ].copy()
    else:
        # If the geometry is already coincident, no reversal is necessary.
        origins_close = np.allclose(
            seg_origin,
            perf_origin,
            atol=2.0,
            rtol=0.0
        )

        directions_close = np.allclose(
            seg_direction,
            perf_direction,
            atol=2e-3,
            rtol=0.0
        )

        if not (
            same_spacing
            and
            origins_close
            and
            directions_close
        ):
            raise RuntimeError(
                "SEGMENT geometry could not be reconciled automatically.\n"
                "The SEGMENT is not on the expected Perf Tomo grid and is "
                "not safely coincident with it.\n"
                f"SEGMENT origin: {seg_origin}\n"
                f"Perf origin   : {perf_origin}\n"
                f"SEGMENT dir   : {seg_direction}\n"
                f"Perf dir      : {perf_direction}"
            )

        print(
            "SEGMENT geometry check: geometry already coincident."
        )
        corrected_array = segment_array.copy()

    # Use the Perf Tomo physical grid for the corrected SEGMENT. This is
    # intentional: the SEGMENT is now represented in the same physical
    # patient-coordinate system as the reconstructed NM image.
    (
        segment_affine,
        segment_origin,
        segment_spacing_xyz,
        segment_direction
    ) = _sitk_geometry_to_affine(
        spect_image
    )

    segment_values = corrected_array

    segment_mask = (
        np.isfinite(segment_values)
        &
        (segment_values > 0)
    )

    print(
        f"SEGMENT positive voxels: "
        f"{np.count_nonzero(segment_mask):,}"
    )
    print(
        f"SEGMENT value range     : "
        f"{np.min(segment_values[segment_mask]) if np.any(segment_mask) else 0:.6g} "
        f"to "
        f"{np.max(segment_values[segment_mask]) if np.any(segment_mask) else 0:.6g}"
    )

    if not np.any(segment_mask):
        raise RuntimeError(
            "The DICOM SEGMENT contains no positive voxels."
        )

    return {
        "data": segment_values,
        "mask": segment_mask,
        "affine": segment_affine,
        "origin": segment_origin,
        "spacing_xyz": segment_spacing_xyz,
        "direction": segment_direction,
        "path": str(segment_path),
        "reverse_z": bool(reverse_z),
    }


# =============================================================================
# APPLY THE SAME SPECT REGISTRATION TRANSFORM TO SEGMENT
# =============================================================================

def transform_segment_for_spect_registration(segment_info, registration_info):
    """
    Put the corrected SEGMENT array into exactly the same registered
    coordinate system used for the SPECT-to-CT registration.

    This is the critical geometry fix. Previously the SEGMENT was resampled
    directly from the raw Perf Tomo geometry to CT, while the SPECT could be
    translated/orientation-rescued during registration. That can clip a
    physically correct ~181 cc SEGMENT down to ~109 cc.

    For normal DICOM-geometry/translation registration, only the registered
    origin changes. For orientation rescue, the exact swap/flip/reverse-Z
    operations used for SPECT are also applied to the SEGMENT.
    """
    data = segment_info["data"].copy()
    label = str(registration_info.get("orientation", "DICOM orientation"))

    if label.startswith("swap="):
        parts = {}
        for item in label.split(","):
            key, value = item.strip().split("=", 1)
            parts[key] = value.lower() == "true"

        if parts.get("swap", False):
            data = data.transpose(0, 2, 1)
        if parts.get("flip_y", False):
            data = data[:, ::-1, :]
        if parts.get("flip_x", False):
            data = data[:, :, ::-1]
        if parts.get("reverse_z", False):
            data = data[::-1, :, :]

    affine = np.asarray(registration_info["affine"], dtype=np.float64)
    origin = np.asarray(registration_info["origin"], dtype=np.float64)

    if data.shape != tuple(registration_info["spect_shape"]):
        raise RuntimeError(
            "SEGMENT/SPECT registration transform produced a shape mismatch: "
            f"SEGMENT {data.shape} vs registered SPECT {registration_info['spect_shape']}"
        )

    mask = np.isfinite(data) & (data > 0)

    return {
        "data": data,
        "mask": mask,
        "affine": affine,
        "origin": origin,
        "orientation": label,
    }


# =============================================================================
# GENERIC VOLUME RESAMPLING TO CT
# =============================================================================

def resample_volume_to_ct(
    volume,
    volume_affine,
    volume_origin,
    ct_shape,
    ct_affine,
    ct_origin,
    order=0
):
    """
    Resample a 3-D volume with patient-coordinate affine geometry to the
    exact CT voxel grid.

    order=0 is used for the binary SEGMENT mask.
    order=1 can be used for continuous SEGMENT values.
    """

    if volume_origin is None:
        raise RuntimeError(
            "Volume origin is required for CT resampling."
        )

    inv_volume_affine = np.linalg.inv(
        volume_affine
    )

    registered = np.zeros(
        ct_shape,
        dtype=FLOAT_DTYPE
    )

    x_indices = np.arange(
        ct_shape[2],
        dtype=np.float64
    )

    y_indices = np.arange(
        ct_shape[1],
        dtype=np.float64
    )

    x_grid, y_grid = np.meshgrid(
        x_indices,
        y_indices,
        indexing="xy"
    )

    ct_xy_xyz = np.column_stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.zeros(
                x_grid.size,
                dtype=np.float64
            ),
        ]
    )

    for z in range(ct_shape[0]):

        ct_xyz = ct_xy_xyz.copy()

        ct_xyz[:, 2] = float(z)

        patient = (
            np.asarray(
                ct_origin,
                dtype=np.float64
            )[None, :]
            +
            ct_xyz @ ct_affine.T
        )

        volume_xyz = (
            patient
            -
            np.asarray(
                volume_origin,
                dtype=np.float64
            )[None, :]
        ) @ inv_volume_affine.T

        coords = np.vstack(
            [
                volume_xyz[:, 2],
                volume_xyz[:, 1],
                volume_xyz[:, 0],
            ]
        )

        sampled = ndimage.map_coordinates(
            volume,
            coords,
            order=order,
            mode="constant",
            cval=0.0
        )

        registered[z] = sampled.reshape(
            (
                ct_shape[1],
                ct_shape[2]
            )
        ).astype(
            FLOAT_DTYPE
        )

    return registered


# =============================================================================
# DICOM SEGMENT -> CT
# =============================================================================

def resample_segment_to_ct(
    segment_info,
    ct_shape,
    ct_affine,
    ct_origin
):
    print_header(
        "RESAMPLING DICOM SEGMENT TO CT"
    )

    segment_mask_ct = resample_volume_to_ct(
        segment_info["mask"].astype(np.float32),
        segment_info["affine"],
        segment_info["origin"],
        ct_shape,
        ct_affine,
        ct_origin,
        order=0
    ) > 0.5

    segment_values_ct = resample_volume_to_ct(
        segment_info["data"],
        segment_info["affine"],
        segment_info["origin"],
        ct_shape,
        ct_affine,
        ct_origin,
        order=1
    )

    if np.count_nonzero(segment_mask_ct) == 0:
        raise RuntimeError(
            "DICOM SEGMENT became empty after resampling to CT."
        )

    segment_mask_ct = (
        segment_mask_ct
        &
        np.isfinite(segment_values_ct)
    )

    segment_values_ct[
        ~segment_mask_ct
    ] = 0.0

    print(
        f"SEGMENT CT voxels      : "
        f"{np.count_nonzero(segment_mask_ct):,}"
    )

    voxel_volume_cm3 = (
        ct_affine[0, 0]
        * ct_affine[1, 1]
        * ct_affine[2, 2]
        /
        1000.0
    )

    segment_volume_cm3 = (
        np.count_nonzero(segment_mask_ct)
        *
        abs(voxel_volume_cm3)
    )

    print(
        f"SEGMENT volume         : "
        f"{segment_volume_cm3:.3f} cm3"
    )

    return (
        segment_mask_ct,
        segment_values_ct
    )


# =============================================================================
# RECONSTRUCTED SPECT GEOMETRY
# =============================================================================

def get_reconstructed_spect_geometry(
    ds
):

    orientation = _as_float_vector(
        getattr(
            ds,
            "ImageOrientationPatient",
            None
        ),
        6
    )

    position = _as_float_vector(
        getattr(
            ds,
            "ImagePositionPatient",
            None
        ),
        3
    )

    grid_offsets = getattr(
        ds,
        "GridFrameOffsetVector",
        None
    )

    if grid_offsets is not None:

        try:

            grid_offsets = np.asarray(
                [
                    float(x)
                    for x in grid_offsets
                ],
                dtype=np.float64
            )

        except Exception:

            grid_offsets = None

    shared = getattr(
        ds,
        "SharedFunctionalGroupsSequence",
        None
    )

    if shared:

        try:

            fg = shared[0]

            plane_orientation = getattr(
                fg,
                "PlaneOrientationSequence",
                None
            )

            if (
                orientation is None
                and
                plane_orientation
            ):

                orientation = _as_float_vector(
                    getattr(
                        plane_orientation[0],
                        "ImageOrientationPatient",
                        None
                    ),
                    6
                )

            plane_position = getattr(
                fg,
                "PlanePositionSequence",
                None
            )

            if (
                position is None
                and
                plane_position
            ):

                position = _as_float_vector(
                    getattr(
                        plane_position[0],
                        "ImagePositionPatient",
                        None
                    ),
                    3
                )

        except Exception:
            pass

    return (
        orientation,
        position,
        grid_offsets
    )


# =============================================================================
# PER-FRAME SPECT GEOMETRY
# =============================================================================

def get_spect_per_frame_geometry(
    ds,
    n_frames
):

    orientations = []
    positions = []

    sequence = getattr(
        ds,
        "PerFrameFunctionalGroupsSequence",
        None
    )

    if sequence is None:
        return None, None

    for fg in sequence:

        orientation = None
        position = None

        try:

            plane_orientation = getattr(
                fg,
                "PlaneOrientationSequence",
                None
            )

            if plane_orientation:

                orientation = _as_float_vector(
                    getattr(
                        plane_orientation[0],
                        "ImageOrientationPatient",
                        None
                    ),
                    6
                )

        except Exception:
            orientation = None

        try:

            plane_position = getattr(
                fg,
                "PlanePositionSequence",
                None
            )

            if plane_position:

                position = _as_float_vector(
                    getattr(
                        plane_position[0],
                        "ImagePositionPatient",
                        None
                    ),
                    3
                )

        except Exception:
            position = None

        orientations.append(
            orientation
        )

        positions.append(
            position
        )

    if len(orientations) != n_frames:

        return None, None

    if all(
        x is None
        for x in orientations
    ):

        orientations = None

    if all(
        x is None
        for x in positions
    ):

        positions = None

    return (
        orientations,
        positions
    )


# =============================================================================
# LOAD SPECT SERIES
# =============================================================================

def load_spect_series(
    series
):

    print_header(
        "LOADING RECONSTRUCTED SPECT"
    )

    datasets = []

    for path in series["files"]:

        try:

            ds = pydicom.dcmread(
                str(path),
                force=True
            )

        except Exception:

            continue

        if not hasattr(
            ds,
            "PixelData"
        ):
            continue

        datasets.append(
            (
                path,
                ds
            )
        )

    if not datasets:

        raise RuntimeError(
            "No SPECT DICOM objects with PixelData were found."
        )

    path0, ds0 = datasets[0]

    data = ds0.pixel_array.astype(
        np.float32
    )

    if data.ndim == 2:

        data = data[
            None,
            ...,
        ]

    elif data.ndim != 3:

        raise RuntimeError(
            "Expected reconstructed SPECT data to be 2-D or 3-D.\n"
            f"Found shape: {data.shape}"
        )

    slope = get_float(
        ds0,
        "RescaleSlope",
        1.0
    )

    intercept = get_float(
        ds0,
        "RescaleIntercept",
        0.0
    )

    data = (
        data * slope
        + intercept
    ).astype(
        FLOAT_DTYPE
    )

    n_frames = data.shape[0]

    # -------------------------------------------------------------------------
    # Pixel spacing
    # -------------------------------------------------------------------------

    pixel_spacing = _as_float_vector(
        getattr(
            ds0,
            "PixelSpacing",
            None
        ),
        2
    )

    orientation_source = (
        "top-level ImageOrientationPatient"
    )

    shared = getattr(
        ds0,
        "SharedFunctionalGroupsSequence",
        None
    )

    if (
        pixel_spacing is None
        and
        shared
    ):

        try:

            pixel_measures = getattr(
                shared[0],
                "PixelMeasuresSequence",
                None
            )

            if pixel_measures:

                pixel_spacing = _as_float_vector(
                    getattr(
                        pixel_measures[0],
                        "PixelSpacing",
                        None
                    ),
                    2
                )

        except Exception:
            pass

    if pixel_spacing is None:

        raise RuntimeError(
            "SPECT PixelSpacing could not be determined."
        )

    row_spacing = float(
        pixel_spacing[0]
    )

    col_spacing = float(
        pixel_spacing[1]
    )

    # -------------------------------------------------------------------------
    # Slice spacing
    # -------------------------------------------------------------------------

    slice_spacing = get_float(
        ds0,
        "SpacingBetweenSlices",
        None
    )

    if slice_spacing is None:

        slice_spacing = get_float(
            ds0,
            "SliceThickness",
            None
        )

    if (
        slice_spacing is None
        and
        shared
    ):

        try:

            pixel_measures = getattr(
                shared[0],
                "PixelMeasuresSequence",
                None
            )

            if pixel_measures:

                slice_spacing = get_float(
                    pixel_measures[0],
                    "SpacingBetweenSlices",
                    None
                )

                if slice_spacing is None:

                    slice_spacing = get_float(
                        pixel_measures[0],
                        "SliceThickness",
                        None
                    )

        except Exception:
            pass

    if slice_spacing is None:

        slice_spacing = 1.0

    slice_spacing = abs(
        float(slice_spacing)
    )

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    (
        iop,
        ipp,
        grid_offsets
    ) = get_reconstructed_spect_geometry(
        ds0
    )

    (
        per_frame_orientations,
        per_frame_positions
    ) = get_spect_per_frame_geometry(
        ds0,
        n_frames
    )

    if (
        per_frame_orientations is not None
        and
        any(
            x is not None
            for x in per_frame_orientations
        )
    ):

        first_orientation = next(
            (
                x
                for x in per_frame_orientations
                if x is not None
            ),
            None
        )

        if first_orientation is not None:

            iop = first_orientation

            orientation_source = (
                "PerFrameFunctionalGroupsSequence"
            )

    if iop is not None:

        row_cos = normalise_vector(
            iop[:3]
        )

        col_cos = normalise_vector(
            iop[3:]
        )

        normal = normalise_vector(
            np.cross(
                row_cos,
                col_cos
            )
        )

    else:

        row_cos = np.array(
            [
                0.0,
                1.0,
                0.0
            ],
            dtype=np.float64
        )

        col_cos = np.array(
            [
                1.0,
                0.0,
                0.0
            ],
            dtype=np.float64
        )

        normal = np.array(
            [
                0.0,
                0.0,
                1.0
            ],
            dtype=np.float64
        )

        orientation_source = (
            "fallback orientation"
        )

    # -------------------------------------------------------------------------
    # Determine frame positions
    # -------------------------------------------------------------------------

    positions = None

    if (
        per_frame_positions is not None
        and
        all(
            x is not None
            for x in per_frame_positions
        )
    ):

        positions = np.asarray(
            per_frame_positions,
            dtype=np.float64
        )

        geometry_source = (
            "per-frame reconstructed-image geometry"
        )

    elif (
        ipp is not None
        and
        grid_offsets is not None
        and
        len(grid_offsets) == n_frames
    ):

        positions = (
            np.asarray(ipp)
            [
                None,
                :
            ]
            +
            grid_offsets[:, None]
            * normal[None, :]
        )

        geometry_source = (
            "ImagePositionPatient + "
            "GridFrameOffsetVector"
        )

    elif ipp is not None:

        positions = np.asarray(
            [
                np.asarray(ipp)
                +
                normal * slice_spacing * i
                for i in range(n_frames)
            ],
            dtype=np.float64
        )

        geometry_source = (
            "ImagePositionPatient + "
            "uniform slice spacing"
        )

    else:

        positions = None

        geometry_source = (
            "automatic CT-lung registration"
        )

    # -------------------------------------------------------------------------
    # Sort frames if physical positions are available
    # -------------------------------------------------------------------------

    if positions is not None:

        projections = np.asarray(
            [
                np.dot(
                    p,
                    normal
                )
                for p in positions
            ],
            dtype=np.float64
        )

        order = np.argsort(
            projections
        )

        data = data[
            order
        ]

        positions = positions[
            order
        ]

        if positions.shape[0] > 1:

            diffs = np.diff(
                projections[order]
            )

            diffs = np.abs(
                diffs
            )

            diffs = diffs[
                diffs > 1e-6
            ]

            if diffs.size:

                median_spacing = float(
                    np.median(
                        diffs
                    )
                )

                if (
                    abs(
                        median_spacing
                        -
                        slice_spacing
                    )
                    >
                    max(
                        0.01,
                        0.05 * slice_spacing
                    )
                ):

                    print()
                    print(
                        "WARNING: SPECT DICOM slice spacing differs "
                        "from the nominal spacing."
                    )

                    print(
                        f"Nominal spacing : "
                        f"{slice_spacing:.6f} mm"
                    )

                    print(
                        f"Measured median : "
                        f"{median_spacing:.6f} mm"
                    )

                slice_spacing = median_spacing

        origin = np.asarray(
            positions[0],
            dtype=np.float64
        )

        image_geometry_available = True

    else:

        origin = None

        image_geometry_available = False

    affine_xyz = np.column_stack(
        [
            col_cos * col_spacing,
            row_cos * row_spacing,
            normal * slice_spacing,
        ]
    )

    frame_uid = getattr(
        ds0,
        "FrameOfReferenceUID",
        None
    )

    print(
        f"SPECT shape              : {data.shape}"
    )

    print(
        f"SPECT voxel spacing (mm) : "
        f"{slice_spacing:.6f}, "
        f"{row_spacing:.6f}, "
        f"{col_spacing:.6f}"
    )

    print(
        f"SPECT origin             : {origin}"
    )

    print(
        f"SPECT geometry source    : "
        f"{geometry_source}"
    )

    print(
        f"SPECT orientation source : "
        f"{orientation_source}"
    )

    print(
        f"SPECT FrameOfReferenceUID: "
        f"{frame_uid}"
    )

    positive = (
        np.isfinite(data)
        &
        (data > 0)
    )

    print(
        f"Positive SPECT voxels    : "
        f"{np.count_nonzero(positive):,}"
    )

    print(
        f"Positive SPECT sum       : "
        f"{np.sum(data[positive], dtype=np.float64):.6e}"
    )

    if np.any(positive):

        print(
            f"Maximum SPECT value      : "
            f"{np.max(data[positive]):.6e}"
        )

    print_memory(
        "SPECT",
        data
    )

    return {
        "data":
            data,

        "affine":
            affine_xyz,

        "origin":
            origin,

        "spacing":
            (
                slice_spacing,
                row_spacing,
                col_spacing
            ),

        "row_cos":
            row_cos,

        "col_cos":
            col_cos,

        "normal":
            normal,

        "frame_uid":
            frame_uid,

        "geometry_source":
            geometry_source,

        "orientation_source":
            orientation_source,

        "image_geometry_available":
            image_geometry_available,
    }


# =============================================================================
# RESAMPLE CT LUNG MASK TO SPECT GRID
# =============================================================================

def resample_ct_lung_to_spect(
    lung_mask,
    ct_affine,
    ct_origin,
    spect_shape,
    spect_affine,
    spect_origin,
    subsample=1
):

    if spect_origin is None:

        raise RuntimeError(
            "SPECT origin is required for resampling."
        )

    subsample = max(
        1,
        int(subsample)
    )

    z_indices = np.arange(
        0,
        spect_shape[0],
        subsample,
        dtype=np.float64
    )

    y_indices = np.arange(
        0,
        spect_shape[1],
        subsample,
        dtype=np.float64
    )

    x_indices = np.arange(
        0,
        spect_shape[2],
        subsample,
        dtype=np.float64
    )

    zz, yy, xx = np.meshgrid(
        z_indices,
        y_indices,
        x_indices,
        indexing="ij"
    )

    spect_xyz = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
            zz.ravel(),
        ]
    )

    patient = (
        np.asarray(
            spect_origin,
            dtype=np.float64
        )[None, :]
        +
        spect_xyz @ spect_affine.T
    )

    inv_ct_affine = np.linalg.inv(
        ct_affine
    )

    ct_xyz = (
        patient
        -
        np.asarray(
            ct_origin,
            dtype=np.float64
        )[None, :]
    ) @ inv_ct_affine.T

    coords_zyx = np.vstack(
        [
            ct_xyz[:, 2],
            ct_xyz[:, 1],
            ct_xyz[:, 0],
        ]
    )

    sampled = ndimage.map_coordinates(
        lung_mask.astype(
            np.float32
        ),
        coords_zyx,
        order=0,
        mode="constant",
        cval=0.0
    )

    output_shape = (
        len(z_indices),
        len(y_indices),
        len(x_indices)
    )

    return (
        sampled.reshape(
            output_shape
        )
        > 0.5
    )


# =============================================================================
# REGISTRATION OVERLAP SCORE
# =============================================================================

def registration_overlap_score(
    spect,
    lung_on_spect,
    activity_mask=None
):

    finite = np.isfinite(
        spect
    )

    if SPECT_POSITIVE_ONLY:

        positive = (
            finite
            &
            (spect > 0)
        )

    else:

        positive = finite

    if activity_mask is not None:

        positive &= activity_mask

    total = float(
        np.sum(
            spect[positive],
            dtype=np.float64
        )
    )

    if total <= 0:

        return 0.0

    inside = (
        positive
        &
        lung_on_spect
    )

    inside_activity = float(
        np.sum(
            spect[inside],
            dtype=np.float64
        )
    )

    return (
        inside_activity
        /
        total
    )


# =============================================================================
# SPECT TRANSLATION OPTIMISATION
# =============================================================================

def optimise_spect_translation(
    spect,
    spect_affine,
    initial_origin,
    ct_affine,
    ct_origin,
    lung_mask,
    spect_spacing,
    initial_translation=None
):

    print_header(
        "SPECT TRANSLATION REGISTRATION"
    )

    if initial_translation is None:

        current_translation = np.zeros(
            3,
            dtype=np.float64
        )

    else:

        current_translation = np.asarray(
            initial_translation,
            dtype=np.float64
        ).copy()

    # -------------------------------------------------------------------------
    # Subsample SPECT for registration only.
    # -------------------------------------------------------------------------

    subsample = max(
        1,
        int(REGISTRATION_SUBSAMPLE)
    )

    if subsample > 1:

        spect_reg = spect[
            ::subsample,
            ::subsample,
            ::subsample
        ]

        spect_affine_reg = (
            spect_affine
            @
            np.diag(
                [
                    subsample,
                    subsample,
                    subsample
                ]
            )
        )

    else:

        spect_reg = spect
        spect_affine_reg = spect_affine

    def evaluate(
        translation
    ):

        test_origin = (
            np.asarray(
                initial_origin,
                dtype=np.float64
            )
            +
            np.asarray(
                translation,
                dtype=np.float64
            )
        )

        lung_on_spect = (
            resample_ct_lung_to_spect(
                lung_mask,
                ct_affine,
                ct_origin,
                spect_reg.shape,
                spect_affine_reg,
                test_origin,
                subsample=1
            )
        )

        score = registration_overlap_score(
            spect_reg,
            lung_on_spect
        )

        return (
            float(score),
            test_origin
        )

    initial_score, _ = evaluate(
        current_translation
    )

    print(
        f"Initial overlap score : "
        f"{initial_score:.6f}"
    )

    levels = [
        (
            "COARSE",
            REGISTRATION_COARSE_STEP_MM,
            REGISTRATION_COARSE_RANGE_MM
        ),
        (
            "FINE",
            REGISTRATION_FINE_STEP_MM,
            REGISTRATION_FINE_RANGE_MM
        ),
        (
            "FINAL",
            REGISTRATION_FINAL_STEP_MM,
            REGISTRATION_FINAL_RANGE_MM
        ),
    ]

    for level_name, step_mm, range_mm in levels:

        print()
        print(
            f"{level_name} registration:"
        )

        print(
            f"    step  = {step_mm:.3f} mm"
        )

        print(
            f"    range = ±{range_mm:.3f} mm"
        )

        improved = True

        iteration = 0

        while improved:

            improved = False

            iteration += 1

            current_score, _ = evaluate(
                current_translation
            )

            best_score = current_score

            best_translation = (
                current_translation.copy()
            )

            for axis in range(3):

                for direction in (
                    -1.0,
                    1.0
                ):

                    trial = (
                        current_translation.copy()
                    )

                    trial[axis] += (
                        direction
                        * step_mm
                    )

                    if (
                        abs(trial[axis])
                        >
                        range_mm
                    ):

                        continue

                    score, _ = evaluate(
                        trial
                    )

                    if score > (
                        best_score
                        + 1e-8
                    ):

                        best_score = score

                        best_translation = (
                            trial
                        )

            if best_score > (
                current_score
                + 1e-8
            ):

                current_translation = (
                    best_translation
                )

                improved = True

            if iteration > 100:
                break

        final_level_score, _ = evaluate(
            current_translation
        )

        print(
            f"    translation = "
            f"{current_translation}"
        )

        print(
            f"    score       = "
            f"{final_level_score:.6f}"
        )

    registered_origin = (
        np.asarray(
            initial_origin,
            dtype=np.float64
        )
        +
        current_translation
    )

    final_score, _ = evaluate(
        current_translation
    )

    print()
    print(
        f"Final SPECT translation (mm): "
        f"{current_translation}"
    )

    print(
        f"Final SPECT origin (mm): "
        f"{registered_origin}"
    )

    print(
        f"Initial overlap: "
        f"{initial_score:.6f}"
    )

    print(
        f"Final overlap: "
        f"{final_score:.6f}"
    )

    return (
        registered_origin,
        current_translation,
        initial_score,
        final_score
    )


# =============================================================================
# INITIAL SPECT ORIGIN FROM LUNG CENTRE
# =============================================================================

def calculate_initial_spect_origin(
    spect_shape,
    spect_affine,
    lung_centre_patient
):

    centre_index_xyz = np.array(
        [
            (spect_shape[2] - 1) / 2.0,
            (spect_shape[1] - 1) / 2.0,
            (spect_shape[0] - 1) / 2.0,
        ],
        dtype=np.float64
    )

    spect_fov_centre_offset = (
        spect_affine
        @
        centre_index_xyz
    )

    initial_origin = (
        np.asarray(
            lung_centre_patient,
            dtype=np.float64
        )
        -
        spect_fov_centre_offset
    )

    return initial_origin


# =============================================================================
# RESAMPLE SPECT TO CT GRID
# =============================================================================

def resample_spect_to_ct(
    spect,
    spect_affine,
    spect_origin,
    ct_shape,
    ct_affine,
    ct_origin
):

    if spect_origin is None:

        raise RuntimeError(
            "SPECT origin is required for CT resampling."
        )

    inv_spect_affine = np.linalg.inv(
        spect_affine
    )

    z_indices = np.arange(
        ct_shape[0],
        dtype=np.float64
    )

    y_indices = np.arange(
        ct_shape[1],
        dtype=np.float64
    )

    x_indices = np.arange(
        ct_shape[2],
        dtype=np.float64
    )

    registered = np.zeros(
        ct_shape,
        dtype=FLOAT_DTYPE
    )

    x_grid, y_grid = np.meshgrid(
        x_indices,
        y_indices,
        indexing="xy"
    )

    ct_xy_xyz = np.column_stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.zeros(
                x_grid.size,
                dtype=np.float64
            ),
        ]
    )

    for z in range(
        ct_shape[0]
    ):

        ct_xyz = ct_xy_xyz.copy()

        ct_xyz[:, 2] = float(
            z
        )

        patient = (
            np.asarray(
                ct_origin,
                dtype=np.float64
            )[None, :]
            +
            ct_xyz @ ct_affine.T
        )

        spect_xyz = (
            patient
            -
            np.asarray(
                spect_origin,
                dtype=np.float64
            )[None, :]
        ) @ inv_spect_affine.T

        coords = np.vstack(
            [
                spect_xyz[:, 2],
                spect_xyz[:, 1],
                spect_xyz[:, 0],
            ]
        )

        sampled = ndimage.map_coordinates(
            spect,
            coords,
            order=1,
            mode="constant",
            cval=0.0
        )

        registered[z] = sampled.reshape(
            (
                ct_shape[1],
                ct_shape[2]
            )
        ).astype(
            FLOAT_DTYPE
        )

    return registered


# =============================================================================
# ORIENTATION RESCUE CANDIDATES
# =============================================================================

def generate_orientation_candidates(
    spect,
    spect_spacing,
    ct_row_cos,
    ct_col_cos,
    ct_normal
):

    """
    Generate plausible reconstructed-SPECT orientation candidates.

    The raw reconstructed SPECT volume is transformed using:

        - in-plane x/y axis swap
        - y flip
        - x flip
        - z reversal

    The transformed volume is then expressed using the CT patient
    coordinate axes.

    This rescue branch is only used when the supplied reconstructed
    SPECT DICOM geometry cannot be reconciled with the CT lung.
    """

    slice_spacing = float(
        spect_spacing[0]
    )

    row_spacing = float(
        spect_spacing[1]
    )

    col_spacing = float(
        spect_spacing[2]
    )

    candidates = []

    for swap_xy in (
        False,
        True
    ):

        for flip_y in (
            False,
            True
        ):

            for flip_x in (
                False,
                True
            ):

                for reverse_z in (
                    False,
                    True
                ):

                    data = spect

                    if swap_xy:

                        data = data.transpose(
                            0,
                            2,
                            1
                        )

                        new_row_spacing = (
                            col_spacing
                        )

                        new_col_spacing = (
                            row_spacing
                        )

                    else:

                        new_row_spacing = (
                            row_spacing
                        )

                        new_col_spacing = (
                            col_spacing
                        )

                    if flip_y:

                        data = data[
                            :,
                            ::-1,
                            :
                        ]

                    if flip_x:

                        data = data[
                            :,
                            :,
                            ::-1
                        ]

                    if reverse_z:

                        data = data[
                            ::-1,
                            :,
                            :
                        ]

                    row_direction = (
                        ct_row_cos
                        *
                        (
                            -1.0
                            if flip_y
                            else 1.0
                        )
                    )

                    col_direction = (
                        ct_col_cos
                        *
                        (
                            -1.0
                            if flip_x
                            else 1.0
                        )
                    )

                    normal_direction = (
                        ct_normal
                        *
                        (
                            -1.0
                            if reverse_z
                            else 1.0
                        )
                    )

                    affine = np.column_stack(
                        [
                            col_direction
                            * new_col_spacing,

                            row_direction
                            * new_row_spacing,

                            normal_direction
                            * slice_spacing,
                        ]
                    )

                    label = (
                        f"swap={swap_xy}, "
                        f"flip_y={flip_y}, "
                        f"flip_x={flip_x}, "
                        f"reverse_z={reverse_z}"
                    )

                    candidates.append(
                        {
                            "data":
                                data,

                            "affine":
                                affine,

                            "spacing":
                                (
                                    slice_spacing,
                                    new_row_spacing,
                                    new_col_spacing
                                ),

                            "label":
                                label,
                        }
                    )

    return candidates


# =============================================================================
# ORIENTATION RESCUE REGISTRATION
# =============================================================================

def select_best_spect_orientation(
    spect_info,
    lung_mask,
    ct_affine,
    ct_origin,
    ct_row_cos,
    ct_col_cos,
    ct_normal,
    lung_centre_patient
):

    print_header(
        "SPECT ORIENTATION RESCUE REGISTRATION"
    )

    raw_spect = spect_info[
        "data"
    ]

    candidates = generate_orientation_candidates(
        raw_spect,
        spect_info["spacing"],
        ct_row_cos,
        ct_col_cos,
        ct_normal
    )

    screened = []

    for index, candidate in enumerate(
        candidates
    ):

        candidate_data = candidate[
            "data"
        ]

        candidate_affine = candidate[
            "affine"
        ]

        initial_origin = (
            calculate_initial_spect_origin(
                candidate_data.shape,
                candidate_affine,
                lung_centre_patient
            )
        )

        subsample = max(
            1,
            int(REGISTRATION_SUBSAMPLE)
        )

        if subsample > 1:

            screening_data = candidate_data[
                ::subsample,
                ::subsample,
                ::subsample
            ]

            screening_affine = (
                candidate_affine
                @
                np.diag(
                    [
                        subsample,
                        subsample,
                        subsample
                    ]
                )
            )

        else:

            screening_data = candidate_data
            screening_affine = candidate_affine

        lung_on_spect = (
            resample_ct_lung_to_spect(
                lung_mask,
                ct_affine,
                ct_origin,
                screening_data.shape,
                screening_affine,
                initial_origin,
                subsample=1
            )
        )

        score = registration_overlap_score(
            screening_data,
            lung_on_spect
        )

        screened.append(
            (
                float(score),
                index,
                initial_origin
            )
        )

    screened.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print()
    print(
        "Initial orientation screening:"
    )

    for rank, (
        score,
        index,
        initial_origin
    ) in enumerate(
        screened
    ):

        print(
            f"    {rank + 1:2d}. "
            f"{candidates[index]['label']} "
            f"score={score:.6f}"
        )

    n_refine = min(
        ORIENTATION_REFINEMENT_CANDIDATES,
        len(screened)
    )

    refined_results = []

    for rank in range(
        n_refine
    ):

        screen_score, index, initial_origin = (
            screened[rank]
        )

        candidate = candidates[
            index
        ]

        print()
        print(
            "-" * 80
        )

        print(
            f"Refining orientation "
            f"{rank + 1}/{n_refine}"
        )

        print(
            candidate["label"]
        )

        (
            registered_origin,
            translation,
            initial_score,
            final_score
        ) = optimise_spect_translation(
            candidate["data"],
            candidate["affine"],
            initial_origin,
            ct_affine,
            ct_origin,
            lung_mask,
            candidate["spacing"]
        )

        refined_results.append(
            {
                "score":
                    final_score,

                "candidate":
                    candidate,

                "origin":
                    registered_origin,

                "translation":
                    translation,

                "initial_score":
                    initial_score,

                "final_score":
                    final_score,
            }
        )

    refined_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = refined_results[0]

    print()
    print(
        "=" * 80
    )

    print(
        "BEST ORIENTATION RESCUE RESULT"
    )

    print(
        f"Orientation : "
        f"{best['candidate']['label']}"
    )

    print(
        f"Final overlap : "
        f"{best['final_score']:.6f}"
    )

    print(
        f"Translation (mm) : "
        f"{best['translation']}"
    )

    print(
        f"Origin (mm) : "
        f"{best['origin']}"
    )

    print(
        "=" * 80
    )

    return {
        "data":
            best["candidate"]["data"],

        "affine":
            best["candidate"]["affine"],

        "spacing":
            best["candidate"]["spacing"],

        "origin":
            best["origin"],

        "translation":
            best["translation"],

        "initial_overlap":
            best["initial_score"],

        "final_overlap":
            best["final_score"],

        "orientation_label":
            best["candidate"]["label"],
    }


# =============================================================================
# SPECT REGISTRATION PIPELINE
# =============================================================================

def register_spect_to_ct(
    spect_info,
    lung_mask,
    ct_affine,
    ct_origin,
    ct_row_cos=None,
    ct_col_cos=None,
    ct_normal=None
):

    spect = spect_info[
        "data"
    ]

    spect_affine = spect_info[
        "affine"
    ]

    right_lung_volume_cm3 = (
        np.count_nonzero(right_lung_mask)
        * ct_voxel_volume_cm3
    )
    left_lung_volume_cm3 = (
        np.count_nonzero(left_lung_mask)
        * ct_voxel_volume_cm3
    )

    print()
    print("FIXED CT LUNG GEOMETRY AUDIT")
    print(f"    Right lung : {right_lung_volume_cm3:.6f} cc")
    print(f"    Left lung  : {left_lung_volume_cm3:.6f} cc")
    print(f"    Whole lung : {initial_lung_volume_cm3:.6f} cc")
    print(f"    Expected right: {EXPECTED_RIGHT_LUNG_VOLUME_CM3:.6f} cc")
    print(f"    Expected left : {EXPECTED_LEFT_LUNG_VOLUME_CM3:.6f} cc")
    print(f"    Expected whole: {EXPECTED_TOTAL_LUNG_VOLUME_CM3:.6f} cc")

    geometry_df = pd.DataFrame([
        ["Right lung", np.count_nonzero(right_lung_mask),
         right_lung_volume_cm3, EXPECTED_RIGHT_LUNG_VOLUME_CM3],
        ["Left lung", np.count_nonzero(left_lung_mask),
         left_lung_volume_cm3, EXPECTED_LEFT_LUNG_VOLUME_CM3],
        ["Whole lung", np.count_nonzero(lung_mask),
         initial_lung_volume_cm3, EXPECTED_TOTAL_LUNG_VOLUME_CM3],
    ], columns=[
        "Region", "Voxels", "Volume_cc", "Expected_volume_cc"
    ])
    geometry_df["Difference_percent"] = (
        100.0
        * (
            geometry_df["Volume_cc"]
            / geometry_df["Expected_volume_cc"]
            - 1.0
        )
    )
    geometry_df.to_csv(
        OUTPUT_DIR / "lung_geometry_reference_audit.csv",
        index=False,
    )

    lung_centre_patient = (
        calculate_lung_physical_centre(
            lung_mask,
            ct_affine,
            ct_origin
        )
    )

    print()
    print(
        f"CT lung physical centre (mm): "
        f"{lung_centre_patient}"
    )

    # =========================================================================
    # CASE 1: RECONSTRUCTED DICOM GEOMETRY AVAILABLE
    # =========================================================================

    if spect_info[
        "image_geometry_available"
    ]:

        print_header(
            "VALIDATING RECONSTRUCTED SPECT DICOM IMAGE GEOMETRY"
        )

        dicom_origin = np.asarray(
            spect_info["origin"],
            dtype=np.float64
        )

        dicom_translation = np.zeros(
            3,
            dtype=np.float64
        )

        dicom_lung_on_spect = (
            resample_ct_lung_to_spect(
                lung_mask,
                ct_affine,
                ct_origin,
                spect.shape,
                spect_affine,
                dicom_origin,
                subsample=REGISTRATION_SUBSAMPLE
            )
        )

        dicom_score = registration_overlap_score(
            spect,
            dicom_lung_on_spect
        )

        print(
            f"DICOM geometry overlap: "
            f"{dicom_score:.6f}"
        )

        # ---------------------------------------------------------------------
        # DICOM geometry already consistent
        # ---------------------------------------------------------------------

        if (
            dicom_score
            >=
            MIN_REGISTRATION_OVERLAP_FRACTION
        ):

            registered_origin = (
                dicom_origin
            )

            translation = (
                dicom_translation
            )

            registration_method = (
                "DICOM reconstructed-image geometry "
                "validated against CT lung"
            )

            initial_score = (
                dicom_score
            )

            final_score = (
                dicom_score
            )

            registration_orientation_label = (
                "DICOM orientation"
            )

            registration_spect = spect
            registration_affine = spect_affine

        # ---------------------------------------------------------------------
        # DICOM geometry exists but requires translation refinement
        # ---------------------------------------------------------------------

        elif AUTO_REGISTER_SPECT:

            print()
            print(
                "DICOM geometry did not meet the lung-overlap "
                "threshold."
            )

            print(
                "Attempting translation refinement while "
                "preserving the DICOM orientation."
            )

            (
                refined_origin,
                refined_translation,
                refined_initial_score,
                refined_final_score
            ) = optimise_spect_translation(
                spect,
                spect_affine,
                dicom_origin,
                ct_affine,
                ct_origin,
                lung_mask,
                spect_info["spacing"]
            )

            if (
                refined_final_score
                >=
                MIN_REGISTRATION_OVERLAP_FRACTION
            ):

                registered_origin = (
                    refined_origin
                )

                translation = (
                    refined_translation
                )

                registration_method = (
                    "DICOM reconstructed-image geometry "
                    "+ translation refinement"
                )

                initial_score = (
                    refined_initial_score
                )

                final_score = (
                    refined_final_score
                )

                registration_orientation_label = (
                    "DICOM orientation"
                )

                registration_spect = spect
                registration_affine = spect_affine

            else:

                print()
                print(
                    "DICOM orientation + translation refinement "
                    "still failed the registration threshold."
                )

                if not DICOM_GEOMETRY_RESCUE:

                    raise RuntimeError(
                        "SPECT DICOM geometry could not be "
                        "reconciled with the CT lung."
                    )

                if (
                    ct_row_cos is None
                    or
                    ct_col_cos is None
                    or
                    ct_normal is None
                ):

                    raise RuntimeError(
                        "CT orientation vectors are required "
                        "for orientation rescue."
                    )

                rescue = (
                    select_best_spect_orientation(
                        spect_info,
                        lung_mask,
                        ct_affine,
                        ct_origin,
                        ct_row_cos,
                        ct_col_cos,
                        ct_normal,
                        lung_centre_patient
                    )
                )

                registered_origin = rescue[
                    "origin"
                ]

                translation = rescue[
                    "translation"
                ]

                initial_score = rescue[
                    "initial_overlap"
                ]

                final_score = rescue[
                    "final_overlap"
                ]

                registration_method = (
                    "DICOM geometry rejected; "
                    "orientation/axis/flip rescue "
                    "+ CT-lung translation optimisation"
                )

                registration_orientation_label = rescue[
                    "orientation_label"
                ]

                registration_spect = rescue[
                    "data"
                ]

                registration_affine = rescue[
                    "affine"
                ]

        else:

            raise RuntimeError(
                "DICOM reconstructed-image geometry failed "
                "registration QC and AUTO_REGISTER_SPECT=False."
            )

    # =========================================================================
    # CASE 2: NO RECONSTRUCTED DICOM GEOMETRY
    # =========================================================================

    else:

        if not AUTO_REGISTER_SPECT:

            raise RuntimeError(
                "Reconstructed SPECT image geometry is incomplete "
                "and AUTO_REGISTER_SPECT=False."
            )

        print_header(
            "RECONSTRUCTED SPECT GEOMETRY INCOMPLETE"
        )

        if (
            ct_row_cos is None
            or
            ct_col_cos is None
            or
            ct_normal is None
        ):

            raise RuntimeError(
                "CT orientation vectors are required for "
                "automatic SPECT registration."
            )

        rescue = (
            select_best_spect_orientation(
                spect_info,
                lung_mask,
                ct_affine,
                ct_origin,
                ct_row_cos,
                ct_col_cos,
                ct_normal,
                lung_centre_patient
            )
        )

        registered_origin = rescue[
            "origin"
        ]

        translation = rescue[
            "translation"
        ]

        initial_score = rescue[
            "initial_overlap"
        ]

        final_score = rescue[
            "final_overlap"
        ]

        registration_method = (
            "CT-lung-centre initialisation + "
            "orientation/axis/flip rescue + "
            "coarse-to-fine translation optimisation"
        )

        registration_orientation_label = rescue[
            "orientation_label"
        ]

        registration_spect = rescue[
            "data"
        ]

        registration_affine = rescue[
            "affine"
        ]

    # =========================================================================
    # RESAMPLE FINAL REGISTERED SPECT TO EXACT CT GRID
    # =========================================================================

    print_header(
        "RESAMPLING REGISTERED SPECT TO CT GRID"
    )

    spect_ct = resample_spect_to_ct(
        registration_spect,
        registration_affine,
        registered_origin,
        lung_mask.shape,
        ct_affine,
        ct_origin
    )

    print_memory(
        "Registered SPECT on CT grid",
        spect_ct
    )

    return {
        "spect_ct":
            spect_ct,

        "origin":
            registered_origin,

        "translation":
            translation,

        "method":
            registration_method,

        "orientation":
            registration_orientation_label,

        "initial_overlap":
            initial_score,

        "final_overlap":
            final_score,

        "affine":
            registration_affine,

        "spect_shape":
            registration_spect.shape,
    }


# =============================================================================
# REGISTRATION SANITY CHECK
# =============================================================================

def spect_registration_sanity_check(
    spect_ct,
    lung_mask
):

    print_header(
        "SPECT REGISTRATION SANITY CHECK"
    )

    finite = np.isfinite(
        spect_ct
    )

    positive = (
        finite
        &
        (
            spect_ct > 0
        )
    )

    positive_lung = (
        positive
        &
        lung_mask
    )

    total_activity = float(
        np.sum(
            spect_ct[positive],
            dtype=np.float64
        )
    )

    lung_activity = float(
        np.sum(
            spect_ct[positive_lung],
            dtype=np.float64
        )
    )

    positive_voxels = np.count_nonzero(
        positive
    )

    positive_lung_voxels = np.count_nonzero(
        positive_lung
    )

    overlap_fraction = (
        lung_activity
        /
        max(
            total_activity,
            1e-30
        )
    )

    print(
        f"Positive SPECT voxels anywhere : "
        f"{positive_voxels:,}"
    )

    print(
        f"Positive SPECT voxels in lung  : "
        f"{positive_lung_voxels:,}"
    )

    print(
        f"SPECT activity anywhere        : "
        f"{total_activity:.6e}"
    )

    print(
        f"SPECT activity in lung         : "
        f"{lung_activity:.6e}"
    )

    print(
        f"Activity overlap fraction      : "
        f"{overlap_fraction:.6f}"
    )

    if positive_voxels == 0:

        raise RuntimeError(
            "Registered SPECT contains no positive activity."
        )

    if positive_lung_voxels == 0:

        raise RuntimeError(
            "Registered SPECT contains no positive voxels "
            "inside the lung mask."
        )

    if (
        overlap_fraction
        <
        MIN_REGISTRATION_OVERLAP_FRACTION
    ):

        raise RuntimeError(
            "SPECT registration failed the lung-overlap "
            "quality-control threshold.\n"
            f"Measured overlap = {overlap_fraction:.4f}\n"
            f"Required overlap = "
            f"{MIN_REGISTRATION_OVERLAP_FRACTION:.4f}"
        )

    print()
    print(
        "REGISTRATION QC PASSED."
    )

    return overlap_fraction


# =============================================================================
# REGISTRATION VISUALISATION
# =============================================================================

def make_registration_qc(
    ct_hu,
    spect_ct,
    lung_mask,
    dose_y90,
    dose_lu177,
    output_path
):

    perfusion = np.where(
        lung_mask,
        spect_ct,
        -np.inf
    )

    z = np.unravel_index(
        np.argmax(
            perfusion
        ),
        perfusion.shape
    )[0]

    ct_slice = ct_hu[z]

    spect_slice = spect_ct[z]

    lung_slice = lung_mask[z]

    spect_display = np.where(
        lung_slice,
        spect_slice,
        np.nan
    )

    y90_slice = np.where(
        lung_slice,
        dose_y90[z],
        np.nan
    )

    lu_slice = np.where(
        lung_slice,
        dose_lu177[z],
        np.nan
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4.5)
    )

    axes[0].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=300
    )

    axes[0].set_title(
        "CT anatomy"
    )

    axes[0].axis(
        "off"
    )

    image1 = axes[1].imshow(
        spect_display,
        cmap="viridis"
    )

    axes[1].set_title(
        "Registered 99mTc-MAA"
    )

    axes[1].axis(
        "off"
    )

    fig.colorbar(
        image1,
        ax=axes[1],
        fraction=0.046,
        pad=0.02
    )

    axes[2].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=300
    )

    image2 = axes[2].imshow(
        spect_display,
        cmap="viridis",
        alpha=0.70
    )

    axes[2].set_title(
        "CT + registered MAA"
    )

    axes[2].axis(
        "off"
    )

    fig.colorbar(
        image2,
        ax=axes[2],
        fraction=0.046,
        pad=0.02
    )

    # -------------------------------------------------------------------------
    # CT + Y-90
    # -------------------------------------------------------------------------

    axes[3].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=300
    )

    image3 = axes[3].imshow(
        y90_slice,
        cmap="magma",
        alpha=0.72,
        vmin=DOSE_VMIN,
        vmax=DOSE_VMAX
    )

    axes[3].set_title(
        "CT + Y-90 dose"
    )

    axes[3].axis(
        "off"
    )

    cbar3 = fig.colorbar(
        image3,
        ax=axes[3],
        fraction=0.046,
        pad=0.02
    )

    cbar3.set_label(
        "Gy"
    )

    # -------------------------------------------------------------------------
    # CT + Lu-177
    # -------------------------------------------------------------------------

    axes[4].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=300
    )

    image4 = axes[4].imshow(
        lu_slice,
        cmap="magma",
        alpha=0.72,
        vmin=DOSE_VMIN,
        vmax=DOSE_VMAX
    )

    axes[4].set_title(
        "CT + Lu-177 dose"
    )

    axes[4].axis(
        "off"
    )

    cbar4 = fig.colorbar(
        image4,
        ax=axes[4],
        fraction=0.046,
        pad=0.02
    )

    cbar4.set_label(
        "Gy"
    )

    fig.suptitle(
        f"CT–SPECT registration and dose fusion — axial slice {z}",
        fontsize=13
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94
        ]
    )

    fig.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_path}"
    )

    return z


# =============================================================================
# PERFUSION MAP
# =============================================================================

def make_perfusion_map(
    ct_hu,
    spect_ct,
    lung_mask,
    output_path
):

    perfusion = np.where(
        lung_mask,
        spect_ct,
        -np.inf
    )

    z = np.unravel_index(
        np.argmax(
            perfusion
        ),
        perfusion.shape
    )[0]

    fig, ax = plt.subplots(
        figsize=(8.0, 7.0)
    )

    ax.imshow(
        ct_hu[z],
        cmap="gray",
        vmin=-1000,
        vmax=300
    )

    display = np.where(
        lung_mask[z],
        spect_ct[z],
        np.nan
    )

    image = ax.imshow(
        display,
        cmap="viridis",
        alpha=0.75
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "Relative MAA SPECT"
    )

    ax.set_title(
        f"Registered 99mTc-MAA perfusion — "
        f"axial slice {z}"
    )

    ax.axis(
        "off"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# =============================================================================
# ACTIVITY DISTRIBUTION
# =============================================================================

def make_activity_distribution(
    spect_ct,
    lung_mask,
    segment_mask,
    segment_values=None
):
    """
    Assign the full prescribed activity to the DICOM SEGMENT.

    Default mode:
        UNIFORM_SEGMENT

    Optional mode:
        SPECT_WEIGHTED_SEGMENT

    In both cases SEGMENT is the hard activity boundary. No activity is
    assigned outside SEGMENT.
    """

    print_header(
        "SEGMENT-CONFINED ACTIVITY DISTRIBUTION"
    )

    segment_mask = (
        segment_mask
        &
        lung_mask
    )

    n_segment = int(
        np.count_nonzero(segment_mask)
    )

    if n_segment == 0:
        raise RuntimeError(
            "DICOM SEGMENT contains no voxels inside the CT lung."
        )

    if ACTIVITY_DISTRIBUTION_MODE == "UNIFORM_SEGMENT":

        weights = np.zeros(
            segment_mask.shape,
            dtype=FLOAT_DTYPE
        )

        weights[
            segment_mask
        ] = 1.0

        print(
            "Activity model: uniform distribution within SEGMENT."
        )

    elif ACTIVITY_DISTRIBUTION_MODE == "SPECT_WEIGHTED_SEGMENT":

        valid = (
            np.isfinite(spect_ct)
            &
            (spect_ct > 0)
        )

        weights = np.zeros(
            segment_mask.shape,
            dtype=FLOAT_DTYPE
        )

        use_mask = (
            segment_mask
            &
            valid
        )

        if not np.any(use_mask):
            raise RuntimeError(
                "SPECT_WEIGHTED_SEGMENT was requested, but no positive "
                "registered SPECT activity exists inside SEGMENT."
            )

        weights[
            use_mask
        ] = spect_ct[
            use_mask
        ]

        print(
            "Activity model: MAA-SPECT-weighted within SEGMENT."
        )

    else:
        raise ValueError(
            "Unknown ACTIVITY_DISTRIBUTION_MODE: "
            f"{ACTIVITY_DISTRIBUTION_MODE}"
        )

    total_weight = float(
        np.sum(
            weights,
            dtype=np.float64
        )
    )

    if total_weight <= 0:
        raise RuntimeError(
            "Total SEGMENT activity weight is zero."
        )

    fractions = (
        weights
        /
        total_weight
    ).astype(
        FLOAT_DTYPE
    )

    initial_activity_Bq = (
        INITIAL_ACTIVITY_GBq
        *
        1.0e9
    )

    activity_Bq = (
        fractions
        *
        initial_activity_Bq
    ).astype(
        FLOAT_DTYPE
    )

    total_activity = float(
        np.sum(
            activity_Bq,
            dtype=np.float64
        )
    )

    outside_segment = (
        np.abs(activity_Bq)
        >
        0
    ) & (~segment_mask)

    if np.any(outside_segment):
        raise RuntimeError(
            "Activity has been assigned outside SEGMENT."
        )

    print(
        f"SEGMENT voxels used      : {n_segment:,}"
    )

    print(
        f"Total SEGMENT weight     : "
        f"{total_weight:.6e}"
    )

    print(
        f"Maximum voxel fraction   : "
        f"{np.max(fractions):.6e}"
    )

    print(
        f"Requested activity       : "
        f"{INITIAL_ACTIVITY_GBq:.6f} GBq"
    )

    print(
        f"Assigned activity        : "
        f"{total_activity / 1e9:.9f} GBq"
    )

    print(
        f"Activity outside SEGMENT: "
        f"{float(np.sum(activity_Bq[~segment_mask])):.6e} Bq"
    )

    print_memory(
        "Activity Bq",
        activity_Bq
    )

    return (
        fractions,
        activity_Bq
    )


# =============================================================================
# DECAY PARAMETERS
# =============================================================================

def decay_parameters(
    isotope
):

    half_life_days = (
        HALF_LIFE_DAYS[
            isotope
        ]
    )

    half_life_s = (
        half_life_days
        *
        86400.0
    )

    lam = (
        LN2
        /
        half_life_s
    )

    treatment_time_s = (
        -np.log(
            1.0 - DECAY_FRACTION
        )
        /
        lam
    )

    treatment_time_days = (
        treatment_time_s
        /
        86400.0
    )

    cumulative_decay_factor_s = (
        DECAY_FRACTION
        /
        lam
    )

    print_header(
        f"{isotope} DECAY PARAMETERS"
    )

    print(
        f"Half-life                : "
        f"{half_life_days:.6f} d"
    )

    print(
        f"Decay constant           : "
        f"{lam:.8e} s^-1"
    )

    print(
        f"Decay fraction           : "
        f"{DECAY_FRACTION:.6f}"
    )

    print(
        f"Treatment duration       : "
        f"{treatment_time_days:.6f} d"
    )

    print(
        f"Treatment duration       : "
        f"{treatment_time_days / half_life_days:.6f} half-lives"
    )

    print(
        f"Cumulative decay factor  : "
        f"{cumulative_decay_factor_s:.6e} s"
    )

    return (
        lam,
        treatment_time_s,
        cumulative_decay_factor_s
    )


# =============================================================================
# GRAVES KERNEL COLUMN DETECTION
# =============================================================================

def normalise_column_name(
    name
):

    text = str(
        name
    )

    text = text.strip().lower()

    text = text.replace(
        "\ufeff",
        ""
    )

    text = text.replace(
        "_",
        " "
    )

    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def identify_radius_column(
    columns
):

    preferred = []

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "outer radius" in n
            and
            "cm" in n
        ):

            preferred.append(
                col
            )

    if preferred:
        return preferred[0]

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "radius" in n
            and
            "cm" in n
        ):

            return col

    for col in columns:

        n = normalise_column_name(
            col
        )

        if "radius" in n:

            return col

    return None


def identify_energy_column(
    columns
):

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "energy deposited" in n
            and
            "mev" in n
        ):

            return (
                col,
                "energy_mev_per_decay"
            )

        if (
            "deposited energy" in n
            and
            "mev" in n
        ):

            return (
                col,
                "energy_mev_per_decay"
            )

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "energy" in n
            and
            "mev" in n
            and
            "dose" not in n
        ):

            return (
                col,
                "energy_mev_per_decay"
            )

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "dose" in n
            and
            "mgy" in n
            and
            "mbq" in n
        ):

            return (
                col,
                "dose_mgy_per_mbq_s"
            )

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "mgy" in n
            and
            (
                "mbq s" in n
                or
                "mbq/s" in n
                or
                "mbq" in n
            )
        ):

            return (
                col,
                "dose_mgy_per_mbq_s"
            )

    for col in columns:

        n = normalise_column_name(
            col
        )

        if (
            "dose" in n
            and
            "gy" in n
            and
            "mbq" in n
        ):

            return (
                col,
                "dose_gy_per_mbq_s"
            )

    return (
        None,
        None
    )


# =============================================================================
# GRAVES KERNEL READER
# =============================================================================

def read_graves_kernel(
    path,
    energy_fraction=KERNEL_ENERGY_FRACTION,
    max_radius_cm=KERNEL_MAX_RADIUS_CM
):

    require_file(
        path,
        "Dose-point kernel"
    )

    print_header(
        f"READING DOSE-POINT KERNEL: "
        f"{path.name}"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        first_lines = []

        for _ in range(8):

            line = f.readline()

            if not line:
                break

            first_lines.append(
                line.rstrip("\n")
            )

    print(
        "Kernel file header:"
    )

    for line in first_lines:

        print(
            f"    {line}"
        )

    df = None

    successful_skiprow = None

    for skiprows in range(
        0,
        8
    ):

        try:

            candidate = pd.read_csv(
                path,
                skiprows=skiprows,
                comment="#"
            )

        except Exception:

            continue

        if candidate.shape[1] < 2:
            continue

        columns = [
            str(c).strip()
            for c in candidate.columns
        ]

        radius_candidate = (
            identify_radius_column(
                columns
            )
        )

        (
            energy_candidate,
            energy_type
        ) = identify_energy_column(
            columns
        )

        if (
            radius_candidate is not None
            and
            energy_candidate is not None
        ):

            df = candidate

            successful_skiprow = skiprows

            break

    if df is None:

        try:

            df = pd.read_csv(
                path,
                engine="python",
                comment="#"
            )

        except Exception as exc:

            raise RuntimeError(
                "Could not read Graves kernel CSV.\n"
                f"File: {path}\n"
                f"Error: {exc}"
            )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    print()
    print(
        "Detected CSV columns:"
    )

    for i, col in enumerate(
        df.columns
    ):

        print(
            f"    [{i}] {col}"
        )

    radius_col = identify_radius_column(
        df.columns
    )

    (
        energy_col,
        energy_type
    ) = identify_energy_column(
        df.columns
    )

    if radius_col is None:

        raise RuntimeError(
            "Could not identify kernel radius column.\n"
            f"File: {path}\n"
            f"Columns: {df.columns.tolist()}"
        )

    if energy_col is None:

        raise RuntimeError(
            "Could not identify kernel energy/dose column.\n"
            f"File: {path}\n"
            f"Columns: {df.columns.tolist()}\n\n"
            "Please provide the first ~20 lines of this file."
        )

    print()
    print(
        f"Detected radius column : {radius_col}"
    )

    print(
        f"Detected dose/energy column : {energy_col}"
    )

    print(
        f"Detected quantity type : {energy_type}"
    )

    radii = pd.to_numeric(
        df[radius_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    raw_quantity = pd.to_numeric(
        df[energy_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    valid = (
        np.isfinite(radii)
        &
        np.isfinite(raw_quantity)
        &
        (radii >= 0)
        &
        (raw_quantity >= 0)
    )

    radii = radii[
        valid
    ]

    raw_quantity = raw_quantity[
        valid
    ]

    order = np.argsort(
        radii
    )

    radii = radii[
        order
    ]

    raw_quantity = raw_quantity[
        order
    ]

    unique_radii = np.unique(
        radii
    )

    if unique_radii.size != radii.size:

        combined_quantity = np.zeros(
            unique_radii.size,
            dtype=np.float64
        )

        for i, radius in enumerate(
            unique_radii
        ):

            combined_quantity[i] = np.sum(
                raw_quantity[
                    radii == radius
                ]
            )

        radii = unique_radii

        raw_quantity = combined_quantity

    if energy_type == (
        "energy_mev_per_decay"
    ):

        shell_energy_mev = (
            raw_quantity.copy()
        )

        quantity_description = (
            "MeV deposited per decay"
        )

    elif energy_type == (
        "dose_mgy_per_mbq_s"
    ):

        shell_dose_mgy_per_mbq_s = (
            raw_quantity
        )

        outer_r = radii

        inner_r = np.zeros_like(
            outer_r
        )

        if outer_r.size > 1:

            inner_r[1:] = (
                outer_r[:-1]
            )

        shell_volume_cm3 = (
            4.0
            /
            3.0
            *
            np.pi
            *
            (
                outer_r ** 3
                -
                inner_r ** 3
            )
        )

        shell_mass_kg = (
            shell_volume_cm3
            *
            WATER_DENSITY_G_CM3
            /
            1000.0
        )

        shell_dose_Gy_per_mbq_s = (
            shell_dose_mgy_per_mbq_s
            *
            1.0e-3
        )

        shell_energy_J_per_decay = (
            shell_dose_Gy_per_mbq_s
            *
            shell_mass_kg
            /
            1.0e6
        )

        shell_energy_mev = (
            shell_energy_J_per_decay
            /
            MEV_TO_J
        )

        quantity_description = (
            "Converted from mGy/(MBq s) "
            "to MeV deposited per decay"
        )

    elif energy_type == (
        "dose_gy_per_mbq_s"
    ):

        outer_r = radii

        inner_r = np.zeros_like(
            outer_r
        )

        if outer_r.size > 1:

            inner_r[1:] = (
                outer_r[:-1]
            )

        shell_volume_cm3 = (
            4.0
            /
            3.0
            *
            np.pi
            *
            (
                outer_r ** 3
                -
                inner_r ** 3
            )
        )

        shell_mass_kg = (
            shell_volume_cm3
            *
            WATER_DENSITY_G_CM3
            /
            1000.0
        )

        shell_energy_J_per_decay = (
            raw_quantity
            *
            shell_mass_kg
            /
            1.0e6
        )

        shell_energy_mev = (
            shell_energy_J_per_decay
            /
            MEV_TO_J
        )

        quantity_description = (
            "Converted from Gy/(MBq s) "
            "to MeV deposited per decay"
        )

    else:

        raise RuntimeError(
            f"Unsupported kernel quantity type: "
            f"{energy_type}"
        )

    valid_energy = (
        np.isfinite(
            shell_energy_mev
        )
        &
        (
            shell_energy_mev >= 0
        )
    )

    radii = radii[
        valid_energy
    ]

    shell_energy_mev = shell_energy_mev[
        valid_energy
    ]

    if radii.size == 0:

        raise RuntimeError(
            "Kernel contains no valid radial bins."
        )

    total_energy_mev = float(
        np.sum(
            shell_energy_mev,
            dtype=np.float64
        )
    )

    if total_energy_mev <= 0:

        raise RuntimeError(
            "Total Graves kernel energy is zero."
        )

    cumulative_energy = np.cumsum(
        shell_energy_mev,
        dtype=np.float64
    )

    cumulative_fraction = (
        cumulative_energy
        /
        total_energy_mev
    )

    cutoff_index = np.searchsorted(
        cumulative_fraction,
        energy_fraction,
        side="left"
    )

    cutoff_index = min(
        cutoff_index,
        len(radii) - 1
    )

    if max_radius_cm is not None:

        radius_indices = np.where(
            radii <= max_radius_cm
        )[0]

        if radius_indices.size == 0:

            raise RuntimeError(
                "KERNEL_MAX_RADIUS_CM is smaller "
                "than the first kernel bin."
            )

        cutoff_index = min(
            cutoff_index,
            radius_indices[-1]
        )

    selected_radii = radii[
        :cutoff_index + 1
    ]

    selected_energy = shell_energy_mev[
        :cutoff_index + 1
    ]

    included_energy_mev = float(
        np.sum(
            selected_energy,
            dtype=np.float64
        )
    )

    included_fraction = (
        included_energy_mev
        /
        total_energy_mev
    )

    i99 = min(
        np.searchsorted(
            cumulative_fraction,
            0.99
        ),
        len(radii) - 1
    )

    i999 = min(
        np.searchsorted(
            cumulative_fraction,
            0.999
        ),
        len(radii) - 1
    )

    i9999 = min(
        np.searchsorted(
            cumulative_fraction,
            0.9999
        ),
        len(radii) - 1
    )

    print()
    print(
        f"Total kernel bins        : "
        f"{len(radii):,}"
    )

    print(
        f"Kernel quantity          : "
        f"{quantity_description}"
    )

    print(
        f"Total deposited energy   : "
        f"{total_energy_mev:.8f} MeV/decay"
    )

    print(
        f"99.0% radius             : "
        f"{radii[i99]:.4f} cm"
    )

    print(
        f"99.9% radius             : "
        f"{radii[i999]:.4f} cm"
    )

    print(
        f"99.99% radius            : "
        f"{radii[i9999]:.4f} cm"
    )

    print(
        f"Selected energy fraction : "
        f"{included_fraction:.8f}"
    )

    print(
        f"Selected water radius    : "
        f"{selected_radii[-1]:.6f} cm"
    )

    return (
        selected_radii,
        selected_energy,
        included_fraction,
        total_energy_mev
    )


# =============================================================================
# DENSITY-SCALED CARTESIAN KERNEL
# =============================================================================

def build_cartesian_kernel(
    radii_cm,
    shell_energy_mev,
    voxel_spacing_mm,
    isotope_name
):

    print_header(
        f"BUILDING REFERENCE-DENSITY GRAVES DPK: "
        f"{isotope_name}"
    )

    if DENSITY_SCALE_KERNEL:

        density_scale = (
            WATER_DENSITY_G_CM3
            /
            LUNG_DENSITY_G_CM3
        )

    else:

        density_scale = 1.0

    scaled_radii_cm = (
        radii_cm
        *
        density_scale
    )

    print(
        "KERNEL DENSITY SCALING"
    )

    print(
        f"Water density            : "
        f"{WATER_DENSITY_G_CM3:.6f} g/cm3"
    )

    print(
        f"Lung density             : "
        f"{LUNG_DENSITY_G_CM3:.6f} g/cm3"
    )

    print(
        f"Physical radius scale    : "
        f"{density_scale:.6f}"
    )

    print(
        "Kernel transport model    : "
        "homogeneous reference-density DPK with global radiological "
        "density scaling"
    )
    print(
        "CT density heterogeneity  : "
        "applied later as voxel-wise local mass correction"
    )
    print(
        "Important                 : "
        "this is NOT a full path-dependent heterogeneous electron transport kernel"
    )

    print(
        f"Water kernel radius      : "
        f"{radii_cm[-1]:.6f} cm"
    )

    print(
        f"Lung kernel radius       : "
        f"{scaled_radii_cm[-1]:.6f} cm"
    )

    dz_cm = (
        voxel_spacing_mm[0]
        /
        10.0
    )

    dy_cm = (
        voxel_spacing_mm[1]
        /
        10.0
    )

    dx_cm = (
        voxel_spacing_mm[2]
        /
        10.0
    )

    max_radius_cm = float(
        scaled_radii_cm[-1]
    )

    nx_half = int(
        np.ceil(
            max_radius_cm
            /
            dx_cm
        )
    )

    ny_half = int(
        np.ceil(
            max_radius_cm
            /
            dy_cm
        )
    )

    nz_half = int(
        np.ceil(
            max_radius_cm
            /
            dz_cm
        )
    )

    nx = (
        2 * nx_half
        +
        1
    )

    ny = (
        2 * ny_half
        +
        1
    )

    nz = (
        2 * nz_half
        +
        1
    )

    print()
    print(
        f"Density-scaled kernel radius : "
        f"{max_radius_cm:.6f} cm"
    )

    print(
        f"Kernel dimensions            : "
        f"({nz}, {ny}, {nx})"
    )

    print(
        f"Voxel spacing                : "
        f"{voxel_spacing_mm[0]:.6f}, "
        f"{voxel_spacing_mm[1]:.6f}, "
        f"{voxel_spacing_mm[2]:.6f} mm"
    )

    estimated_memory_gb = (
        nz
        *
        ny
        *
        nx
        *
        np.dtype(
            FLOAT_DTYPE
        ).itemsize
        /
        (1024 ** 3)
    )

    print(
        f"Estimated kernel memory     : "
        f"{estimated_memory_gb:.3f} GB"
    )

    voxel_volume_cm3 = (
        dx_cm
        *
        dy_cm
        *
        dz_cm
    )

    lung_voxel_mass_g = (
        voxel_volume_cm3
        *
        LUNG_DENSITY_G_CM3
    )

    water_voxel_mass_g = (
        voxel_volume_cm3
        *
        WATER_DENSITY_G_CM3
    )

    print()
    print(
        f"Voxel volume                 : "
        f"{voxel_volume_cm3:.8f} cm3"
    )

    print(
        f"Water voxel mass             : "
        f"{water_voxel_mass_g:.8e} g"
    )

    print(
        f"Lung voxel mass              : "
        f"{lung_voxel_mass_g:.8e} g"
    )

    n_shells = len(
        scaled_radii_cm
    )

    shell_counts = np.zeros(
        n_shells,
        dtype=np.int64
    )

    x = (
        np.arange(
            -nx_half,
            nx_half + 1,
            dtype=np.float64
        )
        *
        dx_cm
    )

    y = (
        np.arange(
            -ny_half,
            ny_half + 1,
            dtype=np.float64
        )
        *
        dy_cm
    )

    xx, yy = np.meshgrid(
        x,
        y,
        indexing="xy"
    )

    r_xy2 = (
        xx * xx
        +
        yy * yy
    )

    print()
    print(
        "Pass 1/2: counting Cartesian voxels..."
    )

    for iz, z_index in enumerate(
        range(
            -nz_half,
            nz_half + 1
        )
    ):

        z_cm = (
            z_index
            *
            dz_cm
        )

        radius = np.sqrt(
            r_xy2
            +
            z_cm * z_cm
        )

        shell_index = np.searchsorted(
            scaled_radii_cm,
            radius,
            side="left"
        )

        valid = (
            shell_index
            <
            n_shells
        )

        if np.any(
            valid
        ):

            counts = np.bincount(
                shell_index[
                    valid
                ].ravel(),
                minlength=n_shells
            )

            shell_counts += counts

    zero_shells = np.where(
        shell_counts == 0
    )[0]

    if zero_shells.size:

        print()
        print(
            f"WARNING: {zero_shells.size} kernel shells "
            "contain no Cartesian voxels."
        )

        print(
            "This is expected only for very fine radial bins "
            "relative to the CT voxel size."
        )

    kernel = np.zeros(
        (
            nz,
            ny,
            nx
        ),
        dtype=FLOAT_DTYPE
    )

    print()
    print(
        "Pass 2/2: constructing density-scaled "
        "Cartesian dose kernel..."
    )

    for iz, z_index in enumerate(
        range(
            -nz_half,
            nz_half + 1
        )
    ):

        z_cm = (
            z_index
            *
            dz_cm
        )

        radius = np.sqrt(
            r_xy2
            +
            z_cm * z_cm
        )

        shell_index = np.searchsorted(
            scaled_radii_cm,
            radius,
            side="left"
        )

        valid = (
            shell_index
            <
            n_shells
        )

        slice_kernel = np.zeros(
            radius.shape,
            dtype=np.float64
        )

        if np.any(
            valid
        ):

            idx = shell_index[
                valid
            ]

            count = shell_counts[
                idx
            ]

            safe = (
                count > 0
            )

            dose_mev_per_g = np.zeros(
                idx.shape,
                dtype=np.float64
            )

            dose_mev_per_g[
                safe
            ] = (
                shell_energy_mev[
                    idx[safe]
                ]
                /
                count[safe]
                /
                lung_voxel_mass_g
            )

            slice_kernel[
                valid
            ] = (
                dose_mev_per_g
                *
                MEV_PER_G_TO_GY
            )

        kernel[iz] = (
            slice_kernel
            .astype(
                FLOAT_DTYPE
            )
        )

    kernel_energy_mev = (
        np.sum(
            kernel.astype(
                np.float64
            ),
            dtype=np.float64
        )
        *
        lung_voxel_mass_g
        /
        MEV_PER_G_TO_GY
    )

    target_energy_mev = float(
        np.sum(
            shell_energy_mev,
            dtype=np.float64
        )
    )

    energy_ratio = (
    kernel_energy_mev
        /
        target_energy_mev
    )

    print()
    print(
        "DENSITY-SCALED KERNEL ENERGY CONSERVATION"
    )

    print(
        f"    Target shell energy : "
        f"{target_energy_mev:.8f} MeV/decay"
    )

    print(
        f"    Cartesian energy     : "
        f"{kernel_energy_mev:.8f} MeV/decay"
    )

    print(
        f"    Ratio                : "
        f"{energy_ratio:.8f}"
    )

    if (
        abs(
            energy_ratio - 1.0
        )
        >
        0.02
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Cartesian kernel energy differs from the "
            "input kernel by >2%."
        )

        print(
            "This is primarily due to discretisation of the "
            "spherical shells on the CT voxel grid."
        )

    print()
    print(
        "Density scaling summary:"
    )

    print(
        f"    Physical radial scale : "
        f"{density_scale:.6f}"
    )

    print(
        f"    Lung voxel mass       : "
        f"{lung_voxel_mass_g:.8e} g"
    )

    print(
        f"    Total kernel energy   : "
        f"{kernel_energy_mev:.8f} MeV/decay"
    )

    print_memory(
        "Density-scaled Cartesian kernel",
        kernel
    )

    return (
        kernel,
        lung_voxel_mass_g,
        max_radius_cm,
        density_scale
    )


# =============================================================================
# =============================================================================
# CARTESIAN DPK NUMERICAL / ENERGY AUDIT
# =============================================================================

def audit_cartesian_kernel(
    kernel,
    shell_energy_mev,
    total_kernel_energy_mev,
    kernel_voxel_mass_g,
    isotope,
):
    """Independent checks of the Cartesian Graves DPK."""

    k = np.asarray(kernel, dtype=np.float64)
    selected_energy = float(np.sum(shell_energy_mev, dtype=np.float64))

    finite_fraction = (
        np.count_nonzero(np.isfinite(k)) / k.size
        if k.size else 0.0
    )
    negative_voxels = int(np.count_nonzero(k < 0.0))
    positive_voxels = int(np.count_nonzero(k > 0.0))

    centre = tuple(int(n // 2) for n in k.shape)
    centre_value = float(k[centre])

    mirror = k[::-1, ::-1, ::-1]
    symmetry_abs = float(np.max(np.abs(k - mirror)))
    symmetry_relative = (
        symmetry_abs / max(float(np.max(np.abs(k))), 1.0e-30)
    )

    cartesian_energy_mev = float(
        np.sum(k, dtype=np.float64)
        * kernel_voxel_mass_g
        / MEV_PER_G_TO_GY
    )
    cartesian_to_selected = (
        cartesian_energy_mev / selected_energy
        if selected_energy > 0 else np.nan
    )
    selected_fraction_full = (
        selected_energy / total_kernel_energy_mev
        if total_kernel_energy_mev > 0 else np.nan
    )

    print()
    print(f"{isotope} CARTESIAN KERNEL AUDIT")
    print(f"    Finite fraction          : {finite_fraction:.12f}")
    print(f"    Negative voxels          : {negative_voxels:,}")
    print(f"    Positive voxels          : {positive_voxels:,}")
    print(f"    Centre kernel value      : {centre_value:.8e} Gy/decay")
    print(f"    Symmetry relative error  : {symmetry_relative:.8e}")
    print(f"    Selected radial energy   : {selected_energy:.8f} MeV/decay")
    print(f"    Full radial energy       : {total_kernel_energy_mev:.8f} MeV/decay")
    print(f"    Selected/full fraction   : {selected_fraction_full:.8f}")
    print(f"    Cartesian energy         : {cartesian_energy_mev:.8f} MeV/decay")
    print(f"    Cartesian/selected ratio : {cartesian_to_selected:.8f}")

    if finite_fraction < 1.0:
        raise RuntimeError(f"{isotope} kernel contains non-finite values.")
    if negative_voxels:
        raise RuntimeError(f"{isotope} kernel contains negative values.")
    if centre_value <= 0.0:
        raise RuntimeError(f"{isotope} kernel centre voxel is not positive.")

    if (
        not np.isfinite(cartesian_to_selected)
        or abs(cartesian_to_selected - 1.0) > 0.02
    ):
        print(
            "WARNING: Cartesian kernel energy differs from selected radial "
            "energy by >2%."
        )

    if symmetry_relative > 1.0e-5:
        print(
            "WARNING: Cartesian kernel symmetry differs by >1e-5 relative."
        )

    return {
        "kernel_finite_fraction": finite_fraction,
        "kernel_negative_voxels": negative_voxels,
        "kernel_positive_voxels": positive_voxels,
        "kernel_centre_Gy_per_decay": centre_value,
        "kernel_symmetry_relative": symmetry_relative,
        "kernel_selected_energy_MeV_per_decay": selected_energy,
        "kernel_full_energy_MeV_per_decay": total_kernel_energy_mev,
        "kernel_selected_fraction_of_full": selected_fraction_full,
        "kernel_cartesian_energy_MeV_per_decay": cartesian_energy_mev,
        "kernel_cartesian_to_selected_ratio": cartesian_to_selected,
    }


# SOURCE CROPPING
# =============================================================================

def crop_source_for_kernel(
    activity_decays,
    spacing_mm,
    kernel_radius_cm
):

    print_header(
        "CROPPING SOURCE FOR CONVOLUTION"
    )

    source_mask = (
        activity_decays > 0
    )

    indices = np.argwhere(
        source_mask
    )

    if indices.size == 0:

        raise RuntimeError(
            "Activity distribution is empty."
        )

    source_min = np.min(
        indices,
        axis=0
    )

    source_max = np.max(
        indices,
        axis=0
    )

    margin = np.ceil(
        kernel_radius_cm
        /
        (
            np.asarray(
                spacing_mm
            )
            /
            10.0
        )
    ).astype(
        int
    )

    margin += (
        EXTRA_CROP_MARGIN_VOXELS
    )

    crop_min = np.maximum(
        0,
        source_min - margin
    )

    crop_max = np.minimum(
        np.asarray(
            activity_decays.shape
        )
        -
        1,
        source_max + margin
    )

    crop_slices = tuple(
        slice(
            int(crop_min[i]),
            int(crop_max[i]) + 1
        )
        for i in range(3)
    )

    source_crop = (
        activity_decays[
            crop_slices
        ]
        .copy()
        .astype(
            FLOAT_DTYPE
        )
    )

    print(
        f"Original source shape : "
        f"{activity_decays.shape}"
    )

    print(
        f"Source bounding box   : "
        f"{source_min} -> {source_max}"
    )

    print(
        f"Crop minimum          : "
        f"{crop_min}"
    )

    print(
        f"Crop maximum          : "
        f"{crop_max}"
    )

    print(
        f"Convolution shape     : "
        f"{source_crop.shape}"
    )

    return (
        source_crop,
        crop_slices
    )


# =============================================================================
# CONVOLUTION
# =============================================================================

def convolve_activity_with_kernel(
    activity_decays,
    kernel,
    spacing_mm,
    kernel_radius_cm
):

    print_header(
        "3-D DENSITY-SCALED DPK CONVOLUTION"
    )

    if CROP_TO_LUNG:

        (
            source_crop,
            crop_slices
        ) = crop_source_for_kernel(
            activity_decays,
            spacing_mm,
            kernel_radius_cm
        )

    else:

        source_crop = (
            activity_decays
        )

        crop_slices = tuple(
            slice(
                0,
                s
            )
            for s in activity_decays.shape
        )

    print(
        "Performing overlap-add convolution..."
    )

    dose_crop = oaconvolve(
        source_crop,
        kernel,
        mode="same"
    ).astype(
        FLOAT_DTYPE
    )

    dose = np.zeros(
        activity_decays.shape,
        dtype=FLOAT_DTYPE
    )

    dose[
        crop_slices
    ] = dose_crop

    del source_crop
    del dose_crop

    gc.collect()

    print_memory(
        "Final dose",
        dose
    )

    return dose


# =============================================================================
# DOSE STATISTICS
# =============================================================================

def calculate_dose_statistics(
    dose,
    lung_mask,
    segment_mask=None
):
    """Calculate descriptive DVH metrics for lung regions."""

    print_header("DOSE STATISTICS")

    finite = np.isfinite(dose)
    lung_values = dose[lung_mask & finite].astype(np.float64)

    if lung_values.size == 0:
        raise RuntimeError("No finite dose values inside lung.")

    segment_values = None
    non_target_values = None

    if segment_mask is not None:
        segment_mask = segment_mask & lung_mask
        segment_values = dose[segment_mask & finite].astype(np.float64)
        non_target_mask = lung_mask & (~segment_mask)
        non_target_values = dose[non_target_mask & finite].astype(np.float64)

        if segment_values.size == 0:
            raise RuntimeError("SEGMENT contains no finite dose values.")

    def stats(values):
        values = np.maximum(values, 0.0)
        return {
            "min_Gy": float(np.min(values)),
            "max_Gy": float(np.max(values)),
            "mean_Gy": float(np.mean(values)),
            "median_Gy": float(np.median(values)),
            "D95_Gy": float(np.percentile(values, 5)),
            "D90_Gy": float(np.percentile(values, 10)),
            "D75_Gy": float(np.percentile(values, 25)),
            "D50_Gy": float(np.percentile(values, 50)),
            "D25_Gy": float(np.percentile(values, 75)),
            "D10_Gy": float(np.percentile(values, 90)),
            "D5_Gy": float(np.percentile(values, 95)),
            "D2_Gy": float(np.percentile(values, 98)),
            "V1_percent": float(np.mean(values >= 1.0) * 100.0),
            "V2_percent": float(np.mean(values >= 2.0) * 100.0),
            "V5_percent": float(np.mean(values >= 5.0) * 100.0),
            "V10_percent": float(np.mean(values >= 10.0) * 100.0),
            "V20_percent": float(np.mean(values >= 20.0) * 100.0),
            "V30_percent": float(np.mean(values >= 30.0) * 100.0),
            "V50_percent": float(np.mean(values >= 50.0) * 100.0),
            "V75_percent": float(np.mean(values >= 75.0) * 100.0),
            "V100_percent": float(np.mean(values >= 100.0) * 100.0),
        }

    lung_stats = stats(lung_values)
    segment_stats = stats(segment_values) if segment_values is not None else None
    non_target_stats = stats(non_target_values) if non_target_values is not None and non_target_values.size else None

    for label, result in [
        ("WHOLE LUNG", lung_stats),
        ("TREATED SEGMENT", segment_stats),
        ("NON-TARGET LUNG", non_target_stats),
    ]:
        if result is None:
            continue
        print()
        print(label)
        for key, value in result.items():
            unit = "%" if key.endswith("_percent") else "Gy"
            print(f"    {key:<12s}: {value:.6g} {unit}")

    return lung_stats, segment_stats, non_target_stats


# =============================================================================
# DVH
# =============================================================================

def make_dvh(
    dose,
    mask,
    title,
    output_path
):

    values = dose[
        mask
        &
        np.isfinite(dose)
    ]

    if values.size == 0:
        return

    values = values.astype(
        np.float64
    )

    max_dose = float(
        np.max(values)
    )

    if max_dose <= 0:
        return

    bins = np.linspace(
        0,
        max_dose,
        500
    )

    hist, edges = np.histogram(
        values,
        bins=bins
    )

    cumulative = np.cumsum(
        hist[::-1]
    )[::-1]

    cumulative_percent = (
        cumulative
        /
        values.size
        *
        100.0
    )

    dose_axis = edges[:-1]

    fig, ax = plt.subplots(
        figsize=(7.0, 5.5)
    )

    ax.plot(
        dose_axis,
        cumulative_percent,
        linewidth=1.2
    )

    ax.set_xlabel(
        "Dose (Gy)"
    )

    ax.set_ylabel(
        "Volume receiving ≥ dose (%)"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        True,
        which="major",
        alpha=0.25
    )

    ax.minorticks_on()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# =============================================================================
# DOSE MAP
# =============================================================================

def _select_hotspot_and_planes(dose, lung_mask):

    masked = np.where(
        lung_mask & np.isfinite(dose),
        dose,
        -np.inf
    )

    if not np.any(np.isfinite(masked)):
        raise RuntimeError("No finite dose values inside lung.")

    hotspot = np.unravel_index(
        np.argmax(masked),
        masked.shape
    )

    return tuple(int(x) for x in hotspot)


def _dose_display_limits(dose, lung_mask):

    values = dose[
        lung_mask & np.isfinite(dose)
    ]

    values = values[values >= 0]

    if values.size == 0:
        return DOSE_VMIN, max(DOSE_VMAX, 1.0)

    vmax = float(np.max(values))
    return DOSE_VMIN, max(DOSE_VMAX, vmax)


def _add_plane_contours(ax, dose_slice, segment_slice, levels):

    finite = np.isfinite(dose_slice)
    positive = finite & (dose_slice > 0)

    if np.any(positive):
        max_dose = float(np.max(dose_slice[positive]))
        valid_levels = [
            float(x) for x in levels
            if 0 < float(x) <= max_dose
        ]

        if valid_levels:
            contours = ax.contour(
                dose_slice,
                levels=valid_levels,
                linewidths=0.8
            )
            ax.clabel(
                contours,
                inline=True,
                fontsize=6,
                fmt=lambda x: f"{x:g} Gy"
            )

    if segment_slice is not None and np.any(segment_slice):
        ax.contour(
            segment_slice.astype(np.float32),
            levels=[0.5],
            linewidths=1.2
        )


def make_dose_map(
    ct_hu,
    dose,
    lung_mask,
    isotope,
    output_path,
    segment_mask=None,
    spacing_mm=None
):
    """
    Three-plane dose map through the same whole-lung maximum-dose voxel.

    Panels:
        axial    : fixed z at hotspot
        coronal  : fixed y at hotspot
        sagittal : fixed x at hotspot

    The dose colour scale is identical in all three planes. SEGMENT is
    shown as a separate contour. Quantitative arrays are never smoothed.
    """

    hotspot = _select_hotspot_and_planes(dose, lung_mask)
    hz, hy, hx = hotspot

    if spacing_mm is None:
        spacing_mm = (1.0, 1.0, 1.0)

    dz, dy, dx = [float(x) for x in spacing_mm]
    vmin, vmax = _dose_display_limits(dose, lung_mask)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16.0, 5.5)
    )

    # Axial: z fixed, display y vertically and x horizontally.
    axial_ct = ct_hu[hz]
    axial_dose = np.where(lung_mask[hz], dose[hz], np.nan)
    axial_seg = segment_mask[hz] if segment_mask is not None else None

    # Coronal: y fixed, display z vertically and x horizontally.
    coronal_ct = ct_hu[:, hy, :]
    coronal_dose = np.where(lung_mask[:, hy, :], dose[:, hy, :], np.nan)
    coronal_seg = segment_mask[:, hy, :] if segment_mask is not None else None

    # Sagittal: x fixed, display z vertically and y horizontally.
    sagittal_ct = ct_hu[:, :, hx]
    sagittal_dose = np.where(lung_mask[:, :, hx], dose[:, :, hx], np.nan)
    sagittal_seg = segment_mask[:, :, hx] if segment_mask is not None else None

    planes = [
        (axes[0], axial_ct, axial_dose, axial_seg, "Axial", hz),
        (axes[1], coronal_ct, coronal_dose, coronal_seg, "Coronal", hy),
        (axes[2], sagittal_ct, sagittal_dose, sagittal_seg, "Sagittal", hx),
    ]

    image = None

    for ax, ct_slice, dose_slice, seg_slice, name, index in planes:

        ax.imshow(
            ct_slice,
            cmap="gray",
            vmin=-1000,
            vmax=300,
            interpolation="bilinear"
        )

        image = ax.imshow(
            dose_slice,
            cmap="magma",
            alpha=0.68,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear"
        )

        _add_plane_contours(
            ax,
            dose_slice,
            seg_slice,
            DOSE_CONTOUR_LEVELS_GY
        )

        # Hotspot crosshair. Pixel coordinates are used only for the marker.
        if name == "Axial":
            ax.plot(hx, hy, marker="+", markersize=9, mew=1.2)
        elif name == "Coronal":
            ax.plot(hx, hz, marker="+", markersize=9, mew=1.2)
        else:
            ax.plot(hy, hz, marker="+", markersize=9, mew=1.2)

        ax.set_title(f"{name} — index {index}")
        ax.axis("off")

    fig.suptitle(
        f"{isotope} dose distribution — three orthogonal planes\n"
        f"Hotspot = {dose[hotspot]:.3g} Gy at voxel (z,y,x)={hotspot}",
        fontsize=12
    )

    cbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Dose (Gy)")

    fig.tight_layout(rect=(0, 0, 0.96, 0.92))
    fig.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)

    return hotspot


# =============================================================================
# SEGMENT CT MAP
# =============================================================================

def make_segment_map(
    ct_hu,
    segment_mask,
    output_path
):
    """Display the DICOM SEGMENT boundary on an axial CT slice.

    Visualization only: the quantitative SEGMENT mask is not modified.
    """

    if segment_mask is None or not np.any(segment_mask):
        print("WARNING: SEGMENT mask is empty; skipping SEGMENT map.")
        return

    indices = np.argwhere(segment_mask)
    z = int(np.clip(np.round(np.mean(indices[:, 0])), 0, ct_hu.shape[0] - 1))

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.imshow(
        ct_hu[z],
        cmap="gray",
        vmin=-1000,
        vmax=300,
        interpolation="bilinear"
    )

    if np.any(segment_mask[z]):
        ax.contour(
            segment_mask[z].astype(np.float32),
            levels=[0.5],
            linewidths=1.2
        )

    ax.set_title(f"DICOM SEGMENT on CT — axial slice {z}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")



def make_isodose_map(
    ct_hu,
    dose,
    lung_mask,
    segment_mask,
    isotope,
    output_path,
    spacing_mm=None
):
    """Backward-compatible wrapper for the three-plane dose map."""

    return make_dose_map(
        ct_hu,
        dose,
        lung_mask,
        isotope,
        output_path,
        segment_mask=segment_mask,
        spacing_mm=spacing_mm
    )


def make_3d_dose_isosurfaces(
    dose,
    lung_mask,
    segment_mask,
    isotope,
    output_path,
    spacing_mm
):
    """
    3-D visualisation of the lung, SEGMENT boundary and selected dose
    isosurfaces. This is display-only; no quantitative dose values are
    modified or resampled for dosimetry.
    """

    finite_lung = lung_mask & np.isfinite(dose)
    indices = np.argwhere(finite_lung)

    if indices.size == 0:
        return

    margin = 2
    z0 = max(0, int(indices[:, 0].min()) - margin)
    z1 = min(dose.shape[0], int(indices[:, 0].max()) + margin + 1)
    y0 = max(0, int(indices[:, 1].min()) - margin)
    y1 = min(dose.shape[1], int(indices[:, 1].max()) + margin + 1)
    x0 = max(0, int(indices[:, 2].min()) - margin)
    x1 = min(dose.shape[2], int(indices[:, 2].max()) + margin + 1)

    d = dose[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
    lung = lung_mask[z0:z1, y0:y1, x0:x1]
    seg = segment_mask[z0:z1, y0:y1, x0:x1]

    voxel_count = d.size
    stride = max(
        1,
        int(np.ceil((voxel_count / MAX_3D_VIS_VOXELS) ** (1.0 / 3.0)))
    )

    if stride > 1:
        d = d[::stride, ::stride, ::stride]
        lung = lung[::stride, ::stride, ::stride]
        seg = seg[::stride, ::stride, ::stride]

    dz, dy, dx = [float(x) for x in spacing_mm]
    vis_spacing = (dz * stride, dy * stride, dx * stride)

    # Dose values outside lung are zero for marching-cubes visualisation.
    d_vis = np.where(lung, np.nan_to_num(d, nan=0.0), 0.0)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    # Lung surface.
    if np.any(lung):
        padded = np.pad(lung.astype(np.float32), 1)
        verts, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=vis_spacing
        )
        verts -= np.array(vis_spacing)
        ax.add_collection3d(
            Poly3DCollection(
                verts[faces],
                alpha=0.08,
                linewidths=0.05
            )
        )

    # SEGMENT surface.
    if np.any(seg):
        padded = np.pad(seg.astype(np.float32), 1)
        verts, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=vis_spacing
        )
        verts -= np.array(vis_spacing)
        ax.add_collection3d(
            Poly3DCollection(
                verts[faces],
                alpha=0.25,
                linewidths=0.15
            )
        )

    max_dose = float(np.nanmax(d_vis))

    for level in DOSE_3D_SURFACE_LEVELS_GY:
        if level <= 0 or level >= max_dose:
            continue

        try:
            padded = np.pad(d_vis, 1)
            verts, faces, _, _ = marching_cubes(
                padded,
                level=float(level),
                spacing=vis_spacing
            )
            verts -= np.array(vis_spacing)

            # Use a transparent surface. Matplotlib's default colour cycle
            # is intentionally retained rather than specifying colours.
            ax.add_collection3d(
                Poly3DCollection(
                    verts[faces],
                    alpha=0.10,
                    linewidths=0.05
                )
            )
        except ValueError:
            continue

    shape = d_vis.shape
    extent_x = shape[2] * vis_spacing[2]
    extent_y = shape[1] * vis_spacing[1]
    extent_z = shape[0] * vis_spacing[0]

    ax.set_xlim(0, extent_x)
    ax.set_ylim(0, extent_y)
    ax.set_zlim(0, extent_z)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(
        f"{isotope} 3-D lung / SEGMENT / dose isosurfaces\n"
        f"Dose maximum = {max_dose:.3g} Gy"
    )

    ax.set_box_aspect((extent_x, extent_y, extent_z))
    ax.view_init(elev=24, azim=-60)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)


def make_combined_dvh(
    dose,
    lung_mask,
    segment_mask,
    isotope,
    output_path
):
    """Create cumulative DVH for whole lung, SEGMENT and non-target lung."""

    regions = {
        "Whole lung": lung_mask,
        "Treated SEGMENT": segment_mask & lung_mask,
        "Non-target lung": lung_mask & (~segment_mask),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    for name, mask in regions.items():
        values = dose[mask & np.isfinite(dose)].astype(np.float64)
        if values.size == 0:
            continue

        values = np.maximum(values, 0.0)
        max_dose = float(np.max(values))
        if max_dose <= 0:
            continue

        bins = np.linspace(0, max_dose, 600)
        hist, edges = np.histogram(values, bins=bins)
        cumulative = np.cumsum(hist[::-1])[::-1]
        cumulative_percent = cumulative / values.size * 100.0

        ax.plot(
            edges[:-1],
            cumulative_percent,
            linewidth=1.2,
            label=name
        )

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume receiving ≥ dose (%)")
    ax.set_title(f"{isotope} cumulative DVH")
    ax.set_ylim(0, 100)
    ax.grid(True, which="major", alpha=0.25)
    ax.minorticks_on()
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SAVE ARRAY
# =============================================================================

def save_array(
    array,
    path
):

    np.save(
        path,
        array
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# TARGET / NON-TARGET ENERGY AUDIT
# =============================================================================

def calculate_region_energy_audit(
    dose,
    density_g_cm3,
    lung_mask,
    segment_mask,
    spacing_mm
):
    """
    Quantify where the calculated dose energy resides within the anatomical
    lung.

    This is the direct test of beta-energy spill-out from the treated SEGMENT.

    The dose field is not clipped at the SEGMENT boundary. Therefore:
        target energy       = SEGMENT ∩ lung
        non-target energy   = lung minus SEGMENT

    Energy is calculated from:
        E = D * m
    with voxel mass obtained from the CT-derived density map.

    This does not attempt to account for energy outside the CT lung volume;
    that residual is reported relative to the total input kernel energy.
    """

    voxel_volume_cm3 = (
        float(spacing_mm[0])
        * float(spacing_mm[1])
        * float(spacing_mm[2])
        / 1000.0
    )

    rho = np.asarray(
        density_g_cm3,
        dtype=np.float64
    )

    rho = np.where(
        np.isfinite(rho) & (rho > 0),
        rho,
        LUNG_DENSITY_G_CM3
    )

    voxel_mass_kg = (
        rho * voxel_volume_cm3 / 1000.0
    )

    finite = np.isfinite(dose)

    target_mask = (
        lung_mask
        &
        segment_mask
    )

    non_target_mask = (
        lung_mask
        &
        (~segment_mask)
    )

    target_energy_J = float(
        np.sum(
            dose[target_mask & finite].astype(np.float64)
            * voxel_mass_kg[target_mask & finite],
            dtype=np.float64
        )
    )

    non_target_energy_J = float(
        np.sum(
            dose[non_target_mask & finite].astype(np.float64)
            * voxel_mass_kg[non_target_mask & finite],
            dtype=np.float64
        )
    )

    lung_energy_J = (
        target_energy_J
        +
        non_target_energy_J
    )

    non_target_fraction_of_lung = (
        non_target_energy_J / lung_energy_J
        if lung_energy_J > 0
        else np.nan
    )

    target_fraction_of_lung = (
        target_energy_J / lung_energy_J
        if lung_energy_J > 0
        else np.nan
    )

    return {
        "target_energy_J": target_energy_J,
        "non_target_energy_J": non_target_energy_J,
        "lung_energy_J": lung_energy_J,
        "target_energy_fraction_of_lung": target_fraction_of_lung,
        "non_target_energy_fraction_of_lung": non_target_fraction_of_lung,
    }



# =============================================================================
# SAVE STATISTICS
# =============================================================================

def save_statistics(
    isotope,
    lung_stats,
    segment_stats,
    non_target_stats,
    treatment_time_days,
    lung_volume_cm3,
    lung_mass_g,
    segment_volume_cm3,
    segment_mass_g,
    non_target_volume_cm3,
    non_target_mass_g,
    energy_audit
):

    rows = []

    region_geometry = {
        "Whole_Lung": (
            lung_volume_cm3,
            lung_mass_g
        ),
        "Treated_SEGMENT": (
            segment_volume_cm3,
            segment_mass_g
        ),
        "Non_Target_Lung": (
            non_target_volume_cm3,
            non_target_mass_g
        ),
    }

    for region_name, stats in [
        ("Whole_Lung", lung_stats),
        ("Treated_SEGMENT", segment_stats),
        ("Non_Target_Lung", non_target_stats),
    ]:
        if stats is None:
            continue

        region_volume_cm3, region_mass_g = region_geometry[
            region_name
        ]

        row = {
            "Isotope": isotope,
            "Region": region_name,
            "Treatment_time_days": treatment_time_days,
            "Initial_activity_GBq": INITIAL_ACTIVITY_GBq,
            "Activity_distribution_mode": ACTIVITY_DISTRIBUTION_MODE,
            "Lung_density_g_cm3": LUNG_DENSITY_G_CM3,
            "Kernel_density_scaling": DENSITY_SCALE_KERNEL,

            # Region-specific geometry. This replaces the previous behaviour
            # where SEGMENT volume/mass were repeated on every row.
            "Region_volume_cm3": region_volume_cm3,
            "Region_mass_g": region_mass_g,

            # Explicit geometry fields retained for unambiguous reporting.
            "Lung_volume_cm3": lung_volume_cm3,
            "Lung_mass_g": lung_mass_g,
            "SEGMENT_volume_cm3": segment_volume_cm3,
            "SEGMENT_mass_g": segment_mass_g,
            "Non_Target_Lung_volume_cm3": non_target_volume_cm3,
            "Non_Target_Lung_mass_g": non_target_mass_g,

            # Reference whole-lung geometry is QA/reporting only.
            "Reference_lung_volume_cm3": EXPECTED_TOTAL_LUNG_VOLUME_CM3,
            "Reference_lung_mass_g": EXPECTED_TOTAL_LUNG_MASS_G,

            "Kernel_selected_energy_J": energy_audit["selected_kernel_energy_J"],
            "Kernel_full_energy_J": energy_audit["full_kernel_energy_J"],
            "Raw_convolution_energy_J": energy_audit["raw_convolution_energy_J"],
            "Raw_convolution_to_selected_ratio": energy_audit["raw_convolution_to_selected_ratio"],
            "Kernel_cartesian_to_selected_ratio": energy_audit["kernel_audit"]["kernel_cartesian_to_selected_ratio"],
            "Kernel_selected_fraction_of_full": energy_audit["kernel_audit"]["kernel_selected_fraction_of_full"],
            "Kernel_symmetry_relative": energy_audit["kernel_audit"]["kernel_symmetry_relative"],
            "Total_cumulative_decays": energy_audit["total_decays"],
            "Kernel_energy_MeV_per_decay": energy_audit["kernel_energy_MeV_per_decay"],
            "Total_kernel_energy_J": energy_audit["total_kernel_energy_J"],
            "Dose_integrated_energy_J": energy_audit["dose_integrated_energy_J"],
            "Dose_energy_fraction_of_kernel": energy_audit["dose_energy_fraction_of_kernel"],
            "Target_SEGMENT_energy_J": energy_audit["target_energy_J"],
            "Non_Target_Lung_energy_J": energy_audit["non_target_energy_J"],
            "Lung_energy_J": energy_audit["lung_energy_J"],
            "Target_energy_fraction_of_lung": energy_audit["target_energy_fraction_of_lung"],
            "Non_Target_energy_fraction_of_lung": energy_audit["non_target_energy_fraction_of_lung"],
        }

        row.update({
            f"Dose_{key}": value
            for key, value in stats.items()
        })

        rows.append(row)

    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / f"{isotope}_dose_statistics.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# =============================================================================
# RUN ONE ISOTOPE
# =============================================================================

def run_isotope(
    isotope,
    activity_Bq,
    lung_mask,
    segment_mask,
    ct_hu,
    spacing_mm,
    density_g_cm3,
    density_scale_factor
):

    print_header(f"RUNNING {isotope} DOSIMETRY")

    (
        decay_constant,
        treatment_time_s,
        cumulative_decay_factor_s
    ) = decay_parameters(isotope)
    activity_decays = (
        activity_Bq * cumulative_decay_factor_s
    ).astype(FLOAT_DTYPE)

    total_decays = float(np.sum(activity_decays, dtype=np.float64))

    print_header(f"{isotope} ENERGY / DECAY AUDIT")
    print(f"Initial activity          : {INITIAL_ACTIVITY_GBq:.6f} GBq")
    print(f"Decay fraction            : {DECAY_FRACTION:.6f}")
    print(f"Total cumulative decays  : {total_decays:.8e}")

    kernel_path = KERNEL_FILES[isotope]

    (
        radii_cm,
        shell_energy_mev,
        included_fraction,
        total_kernel_energy_mev
    ) = read_graves_kernel(kernel_path)

    (
        kernel,
        kernel_voxel_mass_g,
        kernel_radius_cm,
        density_scale
    ) = build_cartesian_kernel(
        radii_cm,
        shell_energy_mev,
        spacing_mm,
        isotope
    )

    kernel_audit = audit_cartesian_kernel(
        kernel,
        shell_energy_mev,
        total_kernel_energy_mev,
        kernel_voxel_mass_g,
        isotope,
    )

    # The source is required to contain exactly the requested activity and
    # must contain no activity outside SEGMENT.
    source_activity_Bq = float(np.sum(activity_Bq, dtype=np.float64))
    source_decays = float(np.sum(activity_decays, dtype=np.float64))

    if not np.isclose(
        source_activity_Bq,
        INITIAL_ACTIVITY_GBq * 1.0e9,
        rtol=1e-6,
        atol=1.0
    ):
        raise RuntimeError(
            f"Activity normalisation failed: {source_activity_Bq:.8e} Bq"
        )

    if np.any(activity_Bq[~segment_mask] != 0):
        raise RuntimeError("Non-zero activity exists outside SEGMENT.")

    dose = convolve_activity_with_kernel(
        activity_decays,
        kernel,
        spacing_mm,
        kernel_radius_cm
    )

    # ------------------------------------------------------------------
    # RAW CONVOLUTION ENERGY CONSERVATION CHECK
    # ------------------------------------------------------------------
    reference_voxel_mass_kg = kernel_voxel_mass_g / 1000.0

    raw_convolution_energy_J = float(
        np.sum(
            dose.astype(np.float64),
            dtype=np.float64
        )
        * reference_voxel_mass_kg
    )

    selected_kernel_energy_J = (
        total_decays
        * float(np.sum(shell_energy_mev, dtype=np.float64))
        * MEV_TO_J
    )

    full_kernel_energy_J = (
        total_decays
        * total_kernel_energy_mev
        * MEV_TO_J
    )

    raw_convolution_to_selected_ratio = (
        raw_convolution_energy_J / selected_kernel_energy_J
        if selected_kernel_energy_J > 0 else np.nan
    )

    print()
    print(f"{isotope} RAW CONVOLUTION ENERGY AUDIT")
    print(f"    Selected kernel energy       : {selected_kernel_energy_J:.8e} J")
    print(f"    Full kernel energy           : {full_kernel_energy_J:.8e} J")
    print(f"    Convolution grid energy      : {raw_convolution_energy_J:.8e} J")
    print(f"    Grid/selected ratio          : {raw_convolution_to_selected_ratio:.8f}")

    if (
        not np.isfinite(raw_convolution_to_selected_ratio)
        or abs(raw_convolution_to_selected_ratio - 1.0) > 0.03
    ):
        print(
            "WARNING: convolution energy differs from the selected kernel "
            "energy by >3%; inspect finite-grid truncation and normalisation."
        )

    # ------------------------------------------------------------------
    # CT-density correction
    # ------------------------------------------------------------------
    # The v14 Graves kernel is constructed using the reference lung density
    # (0.26 g/cm3). Keep that kernel construction unchanged, but convert the
    # resulting voxel dose to the local CT-derived mass by applying:
    #
    #     D_local = D_reference * rho_reference / rho_CT
    #
    # This is the same voxel-wise density correction used in the previous
    # CT-density implementation. The DPK transport/radial construction is not
    # changed here.
    density_for_dose = np.asarray(
        density_g_cm3,
        dtype=np.float64
    ).copy()

    density_for_dose[
        ~np.isfinite(density_for_dose)
    ] = LUNG_DENSITY_G_CM3

    density_for_dose[
        density_for_dose <= 0.0
    ] = LUNG_DENSITY_G_CM3

    correction = np.ones_like(
        dose,
        dtype=np.float64
    )

    correction[lung_mask] = (
        LUNG_DENSITY_G_CM3
        / density_for_dose[lung_mask]
    )

    dose = (
        dose.astype(np.float64)
        * correction
    ).astype(FLOAT_DTYPE)

    # Outside the lung the density correction is not used for statistics.
    # Keeping the complete dose array is important because the DPK is allowed
    # to deposit energy outside the treated SEGMENT.

    # ------------------------------------------------------------------
    # Physical energy audit
    # ------------------------------------------------------------------
    kernel_energy_J_per_decay = (
        total_kernel_energy_mev * MEV_TO_J
    )

    total_kernel_energy_J = (
        total_decays * kernel_energy_J_per_decay
    )

    energy_audit_regions = calculate_region_energy_audit(
        dose,
        density_for_dose,
        lung_mask,
        segment_mask,
        spacing_mm
    )

    dose_energy_J = energy_audit_regions[
        "lung_energy_J"
    ]

    dose_energy_fraction = (
        dose_energy_J / total_kernel_energy_J
        if total_kernel_energy_J > 0 else np.nan
    )

    non_target_energy_J = energy_audit_regions[
        "non_target_energy_J"
    ]

    target_energy_J = energy_audit_regions[
        "target_energy_J"
    ]

    non_target_energy_fraction = energy_audit_regions[
        "non_target_energy_fraction_of_lung"
    ]

    target_energy_fraction = energy_audit_regions[
        "target_energy_fraction_of_lung"
    ]

    print()
    print("TARGET / NON-TARGET ENERGY TRANSFER AUDIT")
    print(
        f"    Treated SEGMENT energy       : "
        f"{target_energy_J:.8e} J"
    )
    print(
        f"    Non-target lung energy       : "
        f"{non_target_energy_J:.8e} J"
    )
    print(
        f"    Total lung energy            : "
        f"{dose_energy_J:.8e} J"
    )
    print(
        f"    Non-target fraction of lung  : "
        f"{100.0 * non_target_energy_fraction:.6f} %"
    )
    print(
        f"    Target fraction of lung     : "
        f"{100.0 * target_energy_fraction:.6f} %"
    )
    print(
        f"    Lung energy / kernel energy  : "
        f"{dose_energy_fraction:.6f}"
    )

       # ------------------------------------------------------------------
    # Quantitative geometry audit
    # ------------------------------------------------------------------
    # IMPORTANT:
    # Whole-lung geometry and SEGMENT geometry are separate.
    #
    # The SEGMENT is the activity source, whereas the CT lung mask is the
    # dose target. Do not use SEGMENT volume as whole-lung volume.
    #
    # spacing_mm is the spacing of the calculation grid in mm.
    # ------------------------------------------------------------------

    voxel_volume_mm3 = (
        float(spacing_mm[0])
        * float(spacing_mm[1])
        * float(spacing_mm[2])
    )

    voxel_volume_cm3 = (
        voxel_volume_mm3 / 1000.0
    )

    # ------------------------------------------------------------------
    # Whole-lung geometry
    # ------------------------------------------------------------------

    lung_voxels = int(
        np.count_nonzero(lung_mask)
    )

    lung_volume_cm3 = (
        lung_voxels * voxel_volume_cm3
    )

    lung_mass_g = float(
        np.sum(
            density_for_dose[lung_mask],
            dtype=np.float64
        )
        * voxel_volume_cm3
    )

    # ------------------------------------------------------------------
    # SEGMENT geometry
    # ------------------------------------------------------------------

    segment_mask_in_lung = (
        segment_mask & lung_mask
    )

    segment_voxels = int(
        np.count_nonzero(segment_mask_in_lung)
    )

    segment_volume_cm3 = (
        segment_voxels * voxel_volume_cm3
    )

    segment_mass_g = float(
        np.sum(
            density_for_dose[segment_mask_in_lung],
            dtype=np.float64
        )
        * voxel_volume_cm3
    )

    # ------------------------------------------------------------------
    # Non-target lung
    # ------------------------------------------------------------------

    non_target_mask = (
        lung_mask & ~segment_mask
    )

    non_target_mask = (
        lung_mask & (~segment_mask)
    )
    non_target_voxels = int(
        np.count_nonzero(non_target_mask)
    )
    non_target_volume_cm3 = (
        non_target_voxels * voxel_volume_cm3
    )
    non_target_mass_g = float(
        np.sum(
            density_for_dose[non_target_mask],
            dtype=np.float64
        )
        * voxel_volume_cm3
    )

    lung_volume_difference_cm3 = (
        lung_volume_cm3 - EXPECTED_TOTAL_LUNG_VOLUME_CM3
    )
    lung_volume_difference_fraction = (
        lung_volume_difference_cm3 / EXPECTED_TOTAL_LUNG_VOLUME_CM3
        if EXPECTED_TOTAL_LUNG_VOLUME_CM3 > 0 else np.nan
    )

    print()
    print("LUNG / SEGMENT GEOMETRY AUDIT")
    print(f"    CT lung voxels             : {lung_voxels:,}")
    print(f"    CT lung volume             : {lung_volume_cm3:.3f} cm3")
    print(f"    CT lung mass               : {lung_mass_g:.3f} g")
    print(f"    Reference lung volume      : {EXPECTED_TOTAL_LUNG_VOLUME_CM3:.3f} cm3")
    print(f"    Reference lung mass        : {EXPECTED_TOTAL_LUNG_MASS_G:.3f} g")
    print(f"    Lung volume difference     : {lung_volume_difference_cm3:+.3f} cm3")
    print(f"    Lung volume difference (%) : {100.0 * lung_volume_difference_fraction:+.2f}%")
    print(f"    SEGMENT voxels             : {segment_voxels:,}")
    print(f"    SEGMENT volume             : {segment_volume_cm3:.3f} cm3")
    print(f"    SEGMENT mass               : {segment_mass_g:.3f} g")
    print(f"    Non-target lung voxels     : {non_target_voxels:,}")
    print(f"    Non-target lung volume     : {non_target_volume_cm3:.3f} cm3")
    print(f"    Non-target lung mass       : {non_target_mass_g:.3f} g")
    print(f"    Lung volume closure        : {segment_volume_cm3 + non_target_volume_cm3:.3f} cm3")

    if abs(lung_volume_difference_fraction) > EXPECTED_TOTAL_LUNG_VOLUME_TOLERANCE_FRACTION:
        print(
            "WARNING: CT-derived lung volume differs from the expected "
            f"{EXPECTED_TOTAL_LUNG_VOLUME_CM3:.2f} cm3 reference by more than "
            f"{100.0 * EXPECTED_TOTAL_LUNG_VOLUME_TOLERANCE_FRACTION:.1f}%."
        )

    segment_upper_bound_Gy = (
        total_kernel_energy_J / (segment_mass_g / 1000.0)
        if segment_mass_g > 0 else np.nan
    )

    print()
    print("DPK ENERGY SANITY CHECK")
    print(f"    Kernel energy                 : {total_kernel_energy_mev:.8f} MeV/decay")
    print(f"    Total kernel energy           : {total_kernel_energy_J:.8e} J")
    print(f"    Dose-integrated energy        : {dose_energy_J:.8e} J")
    print(f"    Dose energy / kernel energy   : {dose_energy_fraction:.6f}")
    print(f"    SEGMENT voxels                : {segment_voxels:,}")
    print(f"    SEGMENT volume                : {segment_volume_cm3:.3f} cm3")
    print(f"    SEGMENT mass                  : {segment_mass_g:.3f} g")
    print(f"    All-kernel-energy SEGMENT UB  : {segment_upper_bound_Gy:.3f} Gy")

    if dose_energy_fraction > 1.05:
        print("WARNING: integrated lung dose energy exceeds input kernel energy.")
    elif dose_energy_fraction < 0:
        print("WARNING: negative integrated dose energy detected.")

    del activity_decays
    del kernel
    gc.collect()

    (
        lung_stats,
        segment_stats,
        non_target_stats
    ) = calculate_dose_statistics(
        dose,
        lung_mask,
        segment_mask_in_lung
    )

    save_array(
        dose,
        OUTPUT_DIR / f"{isotope}_dose_Gy.npy"
    )

    save_array(
        segment_mask.astype(np.uint8),
        OUTPUT_DIR / "SEGMENT_mask_CT.npy"
    )

    # ------------------------------------------------------------------
    # Three-plane dose map and 3-D isodose visualisation
    # ------------------------------------------------------------------
    make_dose_map(
        ct_hu,
        dose,
        lung_mask,
        isotope,
        OUTPUT_DIR / f"{isotope}_dose_3plane_isodose.png",
        segment_mask=segment_mask,
        spacing_mm=spacing_mm
    )

    make_3d_dose_isosurfaces(
        dose,
        lung_mask,
        segment_mask,
        isotope,
        OUTPUT_DIR / f"{isotope}_dose_3D_isosurfaces.png",
        spacing_mm
    )

    make_combined_dvh(
        dose,
        lung_mask,
        segment_mask_in_lung,
        isotope,
        OUTPUT_DIR / f"{isotope}_lung_segment_DVH.png"
    )

    make_dvh(
        dose,
        lung_mask,
        f"{isotope} Whole Lung DVH",
        OUTPUT_DIR / f"{isotope}_whole_lung_DVH.png"
    )

    make_dvh(
        dose,
        segment_mask_in_lung,
        f"{isotope} Treated SEGMENT DVH",
        OUTPUT_DIR / f"{isotope}_SEGMENT_DVH.png"
    )

    non_target_mask = lung_mask & (~segment_mask)

    make_dvh(
        dose,
        non_target_mask,
        f"{isotope} Non-target Lung DVH",
        OUTPUT_DIR / f"{isotope}_non_target_lung_DVH.png"
    )

    energy_audit = {
        "kernel_audit": kernel_audit,
        "selected_kernel_energy_J": float(selected_kernel_energy_J),
        "full_kernel_energy_J": float(full_kernel_energy_J),
        "raw_convolution_energy_J": float(raw_convolution_energy_J),
        "raw_convolution_to_selected_ratio": float(raw_convolution_to_selected_ratio),
        "total_decays": total_decays,
        "kernel_energy_MeV_per_decay": float(total_kernel_energy_mev),
        "total_kernel_energy_J": float(total_kernel_energy_J),
        "dose_integrated_energy_J": float(dose_energy_J),
        "dose_energy_fraction_of_kernel": float(dose_energy_fraction),
        "target_energy_J": float(target_energy_J),
        "non_target_energy_J": float(non_target_energy_J),
        "lung_energy_J": float(dose_energy_J),
        "target_energy_fraction_of_lung": float(target_energy_fraction),
        "non_target_energy_fraction_of_lung": float(non_target_energy_fraction),
    }

    save_statistics(
        isotope,
        lung_stats,
        segment_stats,
        non_target_stats,
        treatment_time_s / 86400.0,
        lung_volume_cm3,
        lung_mass_g,
        segment_volume_cm3,
        segment_mass_g,
        non_target_volume_cm3,
        non_target_mass_g,
        energy_audit
    )

    metadata = {
        "isotope": isotope,
        "initial_activity_GBq": INITIAL_ACTIVITY_GBq,
        "activity_distribution_mode": ACTIVITY_DISTRIBUTION_MODE,
        "activity_volume": "DICOM SEGMENT",
        "half_life_days": HALF_LIFE_DAYS[isotope],
        "decay_fraction": DECAY_FRACTION,
        "treatment_time_days": treatment_time_s / 86400.0,
        "treatment_time_half_lives": treatment_time_s / (HALF_LIFE_DAYS[isotope] * 86400.0),
        "kernel_file": str(kernel_path),
        "kernel_energy_fraction_retained": included_fraction,
        "kernel_radius_water_cm": float(radii_cm[-1]),
        "kernel_radius_lung_cm": float(kernel_radius_cm),
        "kernel_physical_density_scale": float(density_scale),
        "water_density_g_cm3": WATER_DENSITY_G_CM3,
        "lung_reference_density_g_cm3": LUNG_DENSITY_G_CM3,
        "CT_density_map_used": True,
        "CT_density_calibration_scale_factor": float(density_scale_factor),
        "CT_density_min_g_cm3": DENSITY_MIN_G_CM3,
        "CT_density_max_g_cm3": DENSITY_MAX_G_CM3,
        "CT_density_mean_lung_g_cm3": float(np.mean(density_for_dose[lung_mask])),
        "heterogeneous_transport_kernel_used": False,
        "kernel_transport_model": "Reference-density Graves DPK with global density/radiological-distance scaling",
        "lung_density_g_cm3": float(np.mean(density_for_dose[lung_mask])),
        "kernel_total_energy_MeV_per_decay": float(total_kernel_energy_mev),
        "kernel_selected_energy_MeV_per_decay": float(np.sum(shell_energy_mev, dtype=np.float64)),
        "kernel_cartesian_to_selected_ratio": float(kernel_audit["kernel_cartesian_to_selected_ratio"]),
        "kernel_symmetry_relative": float(kernel_audit["kernel_symmetry_relative"]),
        "kernel_centre_Gy_per_decay": float(kernel_audit["kernel_centre_Gy_per_decay"]),
        "selected_kernel_energy_J": float(selected_kernel_energy_J),
        "full_kernel_energy_J": float(full_kernel_energy_J),
        "raw_convolution_energy_J": float(raw_convolution_energy_J),
        "raw_convolution_to_selected_ratio": float(raw_convolution_to_selected_ratio),
        "kernel_voxel_mass_g": float(kernel_voxel_mass_g),
        "total_cumulative_decays": total_decays,
        "total_kernel_energy_J": total_kernel_energy_J,
        "dose_integrated_energy_J": dose_energy_J,
        "dose_energy_fraction_of_kernel": dose_energy_fraction,
        "target_SEGMENT_energy_J": target_energy_J,
        "non_target_lung_energy_J": non_target_energy_J,
        "lung_energy_J": dose_energy_J,
        "target_energy_fraction_of_lung": target_energy_fraction,
        "non_target_energy_fraction_of_lung": non_target_energy_fraction,

        # Quantitative geometry: lung target and SEGMENT source are reported
        # separately.
        "lung_volume_cm3": lung_volume_cm3,
        "lung_mass_g": lung_mass_g,
        "reference_lung_volume_cm3": EXPECTED_TOTAL_LUNG_VOLUME_CM3,
        "reference_lung_mass_g": EXPECTED_TOTAL_LUNG_MASS_G,
        "lung_volume_difference_cm3": lung_volume_difference_cm3,
        "lung_volume_difference_fraction": lung_volume_difference_fraction,
        "SEGMENT_volume_cm3": segment_volume_cm3,
        "SEGMENT_mass_g": segment_mass_g,
        "non_target_lung_volume_cm3": non_target_volume_cm3,
        "non_target_lung_mass_g": non_target_mass_g,
        "SEGMENT_all_kernel_energy_upper_bound_Gy": segment_upper_bound_Gy,
        "method": "3-D Graves beta DPK convolution using the existing reference-density/radiological-distance approximation, with voxel-wise CT-HU-derived density correction for local mass; activity confined to DICOM SEGMENT; not full heterogeneous Monte Carlo transport",
        "MCDB_S_values_used": False,
        "lung_mask_voxels": int(np.count_nonzero(lung_mask)),
        "segment_mask_voxels": segment_voxels,
        "non_target_lung_voxels": int(np.count_nonzero(non_target_mask)),
    }

    metadata_path = OUTPUT_DIR / f"{isotope}_metadata.txt"
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved: {metadata_path}")

    return dose


# =============================================================================
# SAVE REGISTRATION METADATA
# =============================================================================

def save_registration_metadata(
    spect_info,
    registration_info,
    overlap_fraction,
    ct_origin,
    lung_centre
):

    path = (
        OUTPUT_DIR
        /
        "registration_metadata.txt"
    )

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "CT-SPECT REGISTRATION\n"
        )

        f.write(
            "=====================\n\n"
        )

        f.write(
            f"Registration method: "
            f"{registration_info['method']}\n"
        )

        f.write(
            f"Registration orientation: "
            f"{registration_info.get('orientation', 'not recorded')}\n"
        )

        f.write(
            f"SPECT geometry source: "
            f"{spect_info['geometry_source']}\n"
        )

        f.write(
            f"SPECT orientation source: "
            f"{spect_info['orientation_source']}\n"
        )

        f.write(
            f"Initial overlap fraction: "
            f"{registration_info['initial_overlap']}\n"
        )

        f.write(
            f"Final overlap fraction: "
            f"{registration_info['final_overlap']}\n"
        )

        f.write(
            f"Final QC overlap fraction: "
            f"{overlap_fraction}\n"
        )

        f.write(
            f"CT origin (mm): "
            f"{ct_origin}\n"
        )

        f.write(
            f"CT lung centre (mm): "
            f"{lung_centre}\n"
        )

        f.write(
            f"SPECT registered origin (mm): "
            f"{registration_info['origin']}\n"
        )

        f.write(
            f"SPECT translation (mm): "
            f"{registration_info['translation']}\n"
        )

        f.write(
            "\nDetectorInformationSequence was NOT "
            "used as reconstructed SPECT image geometry.\n"
        )

        f.write(
            "\nDOSIMETRY DENSITY MODEL\n"
        )

        f.write(
            "=======================\n"
        )

        f.write(
            f"Water density (g/cm3): "
            f"{WATER_DENSITY_G_CM3}\n"
        )

        f.write(
            f"Lung density (g/cm3): "
            f"{LUNG_DENSITY_G_CM3}\n"
        )

        f.write(
            f"Kernel density scaling enabled: "
            f"{DENSITY_SCALE_KERNEL}\n"
        )

        f.write(
            f"Physical kernel scale factor: "
            f"{WATER_DENSITY_G_CM3 / LUNG_DENSITY_G_CM3}\n"
        )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "LUNG ABLATION DOSIMETRY"
    )

    print(
        "DICOM root:"
    )

    print(
        DICOM_ROOT
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "DOSIMETRY DENSITY MODEL:"
    )

    print(
        f"    Water density : "
        f"{WATER_DENSITY_G_CM3:.3f} g/cm3"
    )

    print(
        f"    Lung density  : "
        f"{LUNG_DENSITY_G_CM3:.3f} g/cm3"
    )

    print(
        f"    Kernel scale  : "
        f"{WATER_DENSITY_G_CM3 / LUNG_DENSITY_G_CM3:.6f}"
    )

    # -------------------------------------------------------------------------
    # Validate paths
    # -------------------------------------------------------------------------

    if not DICOM_ROOT.exists():

        raise FileNotFoundError(
            f"DICOM root does not exist:\n"
            f"{DICOM_ROOT}"
        )

    if not DICOM_ROOT.is_dir():

        raise NotADirectoryError(
            f"DICOM root is not a directory:\n"
            f"{DICOM_ROOT}"
        )

    for isotope, path in KERNEL_FILES.items():

        require_file(
            path,
            f"{isotope} dose-point kernel"
        )

    # -------------------------------------------------------------------------
    # Discover DICOM
    # -------------------------------------------------------------------------

    series_list = discover_dicom_series(
        DICOM_ROOT
    )

    # -------------------------------------------------------------------------
    # CT
    # -------------------------------------------------------------------------

    ct_series = select_ct_series(
        series_list
    )

    (
        ct_hu,
        ct_affine,
        ct_origin,
        ct_dx,
        ct_dy,
        ct_dz,
        ct_row_cos,
        ct_col_cos,
        ct_normal,
        ct_frame_uid,
    ) = load_ct_series(
        ct_series
    )

    spacing_mm = (
        ct_dz,
        ct_dy,
        ct_dx
    )

    # -------------------------------------------------------------------------
    # Lung segmentation
    # -------------------------------------------------------------------------

    lung_mask = make_lung_mask(
        ct_hu
    )

    # -------------------------------------------------------------------------
    # FIXED ANATOMICAL GEOMETRY — SAME FOR Y-90 AND Lu-177
    # -------------------------------------------------------------------------
    lung_mask, right_lung_mask, left_lung_mask = (
        correct_lung_mask_to_reference_geometry(
            ct_hu,
            lung_mask,
            ct_affine,
            abs(np.linalg.det(ct_affine)) / 1000.0,
        )
    )

    save_array(
        lung_mask.astype(
            np.uint8
        ),
        OUTPUT_DIR
        /
        "lung_mask.npy"
    )

    save_array(
        right_lung_mask.astype(np.uint8),
        OUTPUT_DIR / "right_lung_mask.npy"
    )

    save_array(
        left_lung_mask.astype(np.uint8),
        OUTPUT_DIR / "left_lung_mask.npy"
    )

    # -------------------------------------------------------------------------
    # CT HU -> voxel-wise density map
    # -------------------------------------------------------------------------
    density_g_cm3, density_scale_factor = hu_to_density(
        ct_hu,
        lung_mask,
        LUNG_DENSITY_G_CM3
    )

    save_array(
        ct_hu.astype(FLOAT_DTYPE),
        OUTPUT_DIR / "CT_HU.npy"
    )

    save_array(
        density_g_cm3.astype(FLOAT_DTYPE),
        OUTPUT_DIR / "CT_density_g_cm3.npy"
    )

    voxel_volume_cm3_density = (
        abs(np.linalg.det(ct_affine)) / 1000.0
    )
    voxel_mass_g_map = (
        density_g_cm3
        * voxel_volume_cm3_density
    )
    save_array(
        voxel_mass_g_map.astype(FLOAT_DTYPE),
        OUTPUT_DIR / "CT_voxel_mass_g.npy"
    )

    save_density_figures(
        ct_hu,
        density_g_cm3,
        lung_mask
    )

    print()
    print("CT DENSITY MAP AUDIT")
    print(
        f"    Mean lung density        : "
        f"{np.mean(density_g_cm3[lung_mask]):.6f} g/cm3"
    )
    print(
        f"    Lung density min         : "
        f"{np.min(density_g_cm3[lung_mask]):.6f} g/cm3"
    )
    print(
        f"    Lung density max         : "
        f"{np.max(density_g_cm3[lung_mask]):.6f} g/cm3"
    )

    ct_voxel_volume_cm3 = (
        abs(np.linalg.det(ct_affine)) / 1000.0
    )
    initial_lung_volume_cm3 = (
        np.count_nonzero(lung_mask)
        *
        ct_voxel_volume_cm3
    )
    initial_lung_mass_g = float(
        np.sum(
            density_g_cm3[lung_mask],
            dtype=np.float64
        )
        * ct_voxel_volume_cm3
    )

    print()
    print("CT LUNG GEOMETRY")
    print(
        f"    Lung voxels             : {np.count_nonzero(lung_mask):,}"
    )
    print(
        f"    Voxel volume            : {ct_voxel_volume_cm3:.6f} cm3"
    )
    print(
        f"    CT lung volume          : {initial_lung_volume_cm3:.3f} cm3"
    )
    print(
        f"    CT lung mass            : {initial_lung_mass_g:.3f} g"
    )
    print(
        f"    Reference lung volume   : {EXPECTED_TOTAL_LUNG_VOLUME_CM3:.3f} cm3"
    )
    print(
        f"    Reference lung mass     : {EXPECTED_TOTAL_LUNG_MASS_G:.3f} g"
    )

    initial_lung_volume_difference_fraction = (
        initial_lung_volume_cm3 / EXPECTED_TOTAL_LUNG_VOLUME_CM3 - 1.0
        if EXPECTED_TOTAL_LUNG_VOLUME_CM3 > 0 else np.nan
    )

    if (
        np.isfinite(initial_lung_volume_difference_fraction)
        and
        abs(initial_lung_volume_difference_fraction)
        > EXPECTED_TOTAL_LUNG_VOLUME_TOLERANCE_FRACTION
    ):
        message = (
            f"CT lung volume {initial_lung_volume_cm3:.3f} cm3 differs "
            f"from the approximate reference {EXPECTED_TOTAL_LUNG_VOLUME_CM3:.3f} cm3 "
            f"by {100.0 * initial_lung_volume_difference_fraction:+.2f}%."
        )
        print("WARNING: " + message)
        if FAIL_ON_LUNG_VOLUME_QA:
            raise RuntimeError(message)

    lung_centre_patient = (
        calculate_lung_physical_centre(
            lung_mask,
            ct_affine,
            ct_origin
        )
    )

    print()
    print(
        f"CT lung physical centre (mm): "
        f"{lung_centre_patient}"
    )

    # -------------------------------------------------------------------------
    # SPECT
    # -------------------------------------------------------------------------

    spect_series = select_spect_series(
        series_list
    )

    # -------------------------------------------------------------------------
    # DICOM SEGMENT
    # -------------------------------------------------------------------------

    if not USE_DICOM_SEGMENT:
        raise RuntimeError(
            "This version is configured for DICOM SEGMENT-based dosimetry. "
            "Set USE_DICOM_SEGMENT=True."
        )

    segment_series = select_segment_series(
        series_list
    )

    segment_info = load_and_align_dicom_segment(
        segment_series,
        spect_series
    )

    # Audit the source SEGMENT before any CT resampling. For this dataset this
    # should be approximately 181.15 cm3 (1644 voxels at ~4.7952 mm isotropic).
    segment_voxel_volume_cm3 = (
        abs(np.linalg.det(segment_info["affine"])) / 1000.0
    )
    source_segment_volume_cm3 = (
        np.count_nonzero(segment_info["mask"])
        * segment_voxel_volume_cm3
    )
    print()
    print("SEGMENT GEOMETRY AUDIT — SOURCE / PERF TOMO GRID")
    print(f"    Positive voxels         : {np.count_nonzero(segment_info['mask']):,}")
    print(f"    Voxel volume            : {segment_voxel_volume_cm3:.6f} cm3")
    print(f"    Physical SEGMENT volume: {source_segment_volume_cm3:.3f} cm3")
    print(f"    Expected SEGMENT volume: {EXPECTED_SEGMENT_VOLUME_CM3:.3f} cm3")

    # Do NOT resample the SEGMENT to CT yet. The SPECT registration below may
    # apply a translation or orientation rescue; the SEGMENT must follow that
    # exact transform.

    # -------------------------------------------------------------------------
    # Load reconstructed SPECT
    # -------------------------------------------------------------------------

    spect_info = load_spect_series(
        spect_series
    )

    spect = spect_info[
        "data"
    ]

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    registration_info = (
        register_spect_to_ct(
            spect_info,
            lung_mask,
            ct_affine,
            ct_origin,
            ct_row_cos,
            ct_col_cos,
            ct_normal
        )
    )

    spect_ct = registration_info[
        "spect_ct"
    ]

    # -------------------------------------------------------------------------
    # Apply the SAME registration transform to SEGMENT, then resample to CT
    # -------------------------------------------------------------------------

    registered_segment_info = transform_segment_for_spect_registration(
        segment_info,
        registration_info
    )

    registered_segment_volume_cm3 = (
        np.count_nonzero(registered_segment_info["mask"])
        * abs(np.linalg.det(registered_segment_info["affine"])) / 1000.0
    )

    print()
    print("SEGMENT GEOMETRY AUDIT — AFTER SPECT REGISTRATION")
    print(f"    Registration orientation: {registered_segment_info['orientation']}")
    print(f"    Positive voxels         : {np.count_nonzero(registered_segment_info['mask']):,}")
    print(f"    Physical SEGMENT volume: {registered_segment_volume_cm3:.3f} cm3")

    (
        segment_mask,
        segment_values
    ) = resample_segment_to_ct(
        registered_segment_info,
        ct_hu.shape,
        ct_affine,
        ct_origin
    )

    segment_ct_volume_cm3 = (
        np.count_nonzero(segment_mask)
        * abs(np.linalg.det(ct_affine)) / 1000.0
    )
    segment_ct_in_lung_volume_cm3 = (
        np.count_nonzero(segment_mask & lung_mask)
        * abs(np.linalg.det(ct_affine)) / 1000.0
    )

    print()
    print("SEGMENT GEOMETRY AUDIT — FINAL CT GRID")
    print(f"    CT SEGMENT voxels      : {np.count_nonzero(segment_mask):,}")
    print(f"    CT SEGMENT volume      : {segment_ct_volume_cm3:.3f} cm3")
    print(f"    SEGMENT ∩ lung volume : {segment_ct_in_lung_volume_cm3:.3f} cm3")
    print(f"    Expected SEGMENT volume: {EXPECTED_SEGMENT_VOLUME_CM3:.3f} cm3")
    segment_volume_retention = (
        segment_ct_volume_cm3 / source_segment_volume_cm3
        if source_segment_volume_cm3 > 0 else 0.0
    )

    print(
        f"    Volume retention       : "
        f"{100.0 * segment_volume_retention:.2f}%"
    )

    if segment_ct_volume_cm3 < EXPECTED_SEGMENT_VOLUME_CM3 * (1.0 - EXPECTED_SEGMENT_VOLUME_TOLERANCE_FRACTION):
        message = (
            "Final CT SEGMENT volume is substantially below the expected "
            "physical SEGMENT volume. The mask is NOT artificially dilated; "
            "inspect registration/QC outputs."
        )
        print("WARNING: " + message)
        if FAIL_ON_SEGMENT_VOLUME_LOSS:
            raise RuntimeError(message)

    segment_lung_retention = (
        segment_ct_in_lung_volume_cm3 / segment_ct_volume_cm3
        if segment_ct_volume_cm3 > 0 else 0.0
    )

    print(
        f"    SEGMENT-in-lung retention: "
        f"{100.0 * segment_lung_retention:.2f}%"
    )

    if segment_lung_retention < 0.95:
        print(
            "WARNING: More than 5% of the registered SEGMENT lies outside "
            "the CT-derived lung mask."
        )

    save_array(
        segment_mask.astype(np.uint8),
        OUTPUT_DIR / "SEGMENT_mask_CT.npy"
    )

    save_array(
        segment_values.astype(FLOAT_DTYPE),
        OUTPUT_DIR / "SEGMENT_values_CT.npy"
    )

    save_array(
        segment_info["data"].astype(FLOAT_DTYPE),
        OUTPUT_DIR / "SEGMENT_corrected_to_PerfTomo.npy"
    )

    with open(OUTPUT_DIR / "SEGMENT_geometry_metadata.txt", "w") as f:
        f.write("DICOM SEGMENT DOSIMETRY GEOMETRY\n")
        f.write("=================================\n\n")
        f.write(f"SEGMENT source: {segment_info['path']}\n")
        f.write(f"Initial reverse Z correction: {segment_info['reverse_z']}\n")
        f.write(f"Source SEGMENT positive voxels: {np.count_nonzero(segment_info['mask'])}\n")
        f.write(f"Source SEGMENT volume (cm3): {source_segment_volume_cm3:.6f}\n")
        f.write(f"Registration orientation: {registered_segment_info['orientation']}\n")
        f.write(f"Registered SEGMENT volume (cm3): {registered_segment_volume_cm3:.6f}\n")
        f.write(f"SEGMENT CT voxels: {np.count_nonzero(segment_mask)}\n")
        f.write(f"SEGMENT CT volume (cm3): {segment_ct_volume_cm3:.6f}\n")
        f.write(f"SEGMENT CT volume retention (%): {100.0 * segment_volume_retention:.3f}\n")
        f.write(f"SEGMENT-in-lung retention (%): {100.0 * segment_lung_retention:.3f}\n")
        f.write(f"SEGMENT intersected with CT lung volume (cm3): {segment_ct_in_lung_volume_cm3:.6f}\n")
        f.write(f"CT voxel volume (cm3): {abs(np.linalg.det(ct_affine)) / 1000.0:.9f}\n")
        f.write(f"Expected treated SEGMENT volume (cm3): {EXPECTED_SEGMENT_VOLUME_CM3:.6f}\n")
        f.write("\nThe SEGMENT is transformed using the exact SPECT registration before CT resampling.\n")
        f.write("No dilation or forced volume correction is applied.\n")

    # -------------------------------------------------------------------------
    # Final registration QC
    # -------------------------------------------------------------------------

    overlap_fraction = (
        spect_registration_sanity_check(
            spect_ct,
            lung_mask
        )
    )

    # -------------------------------------------------------------------------
    # Save registered SPECT
    # -------------------------------------------------------------------------

    save_array(
        spect_ct,
        OUTPUT_DIR
        /
        "registered_MAA_SPECT.npy"
    )

    # -------------------------------------------------------------------------
    # Save registration metadata
    # -------------------------------------------------------------------------

    save_registration_metadata(
        spect_info,
        registration_info,
        overlap_fraction,
        ct_origin,
        lung_centre_patient
    )

    # -------------------------------------------------------------------------
    # Perfusion map
    # -------------------------------------------------------------------------

    make_perfusion_map(
        ct_hu,
        spect_ct,
        lung_mask,
        OUTPUT_DIR
        /
        "registered_MAA_perfusion.png"
    )

    # -------------------------------------------------------------------------
    # SEGMENT map
    # -------------------------------------------------------------------------

    make_segment_map(
        ct_hu,
        segment_mask,
        OUTPUT_DIR
        /
        "SEGMENT_on_CT.png"
    )

    # -------------------------------------------------------------------------
    # Activity distribution
    # -------------------------------------------------------------------------

    (
        activity_fraction,
        activity_Bq
    ) = make_activity_distribution(
        spect_ct,
        lung_mask,
        segment_mask,
        segment_values
    )

    save_array(
        activity_fraction,
        OUTPUT_DIR
        /
        "activity_fraction.npy"
    )

    save_array(
        activity_Bq,
        OUTPUT_DIR
        /
        "initial_activity_Bq.npy"
    )

    # -------------------------------------------------------------------------
    # Y-90
    # -------------------------------------------------------------------------

    dose_y90 = run_isotope(
        "Y90",
        activity_Bq,
        lung_mask,
        segment_mask,
        ct_hu,
        spacing_mm,
        density_g_cm3,
        density_scale_factor
    )

    # -------------------------------------------------------------------------
    # Lu-177
    # -------------------------------------------------------------------------

    dose_lu177 = run_isotope(
        "Lu177",
        activity_Bq,
        lung_mask,
        segment_mask,
        ct_hu,
        spacing_mm,
        density_g_cm3,
        density_scale_factor
    )


    # -------------------------------------------------------------------------
    # Combined registration/dose QC figure
    # -------------------------------------------------------------------------

    make_registration_qc(
        ct_hu,
        spect_ct,
        lung_mask,
        dose_y90,
        dose_lu177,
        OUTPUT_DIR
        /
        "CT_SPECT_DOSE_registration_QC.png"
    )

    # -------------------------------------------------------------------------
    # Release large arrays
    # -------------------------------------------------------------------------

    del dose_y90
    del dose_lu177
    del activity_Bq
    del activity_fraction
    del spect_ct
    del spect

    gc.collect()

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_header(
        "ANALYSIS COMPLETE"
    )

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Important registration outputs:"
    )

    print(
        "    registered_MAA_SPECT.npy"
    )

    print(
        "    registered_MAA_perfusion.png"
    )

    print(
        "    registration_metadata.txt"
    )

    print(
        "    CT_SPECT_DOSE_registration_QC.png"
    )

    print()
    print(
        "CT density outputs:"
    )

    print(
        "    CT_HU.npy"
    )

    print(
        "    CT_density_g_cm3.npy"
    )

    print(
        "    CT_voxel_mass_g.npy"
    )

    print(
        "    density/CT_density_map.png"
    )

    print(
        "    density/CT_density_histogram.png"
    )

    print()
    print(
        "Dosimetry outputs:"
    )

    print(
        "    activity_fraction.npy"
    )

    print(
        "    initial_activity_Bq.npy"
    )

    print(
        "    Y90_dose_Gy.npy"
    )

    print(
        "    SEGMENT_mask_CT.npy"
    )

    print(
        "    SEGMENT_values_CT.npy"
    )

    print(
        "    Y90_dose_Gy.npy"
    )

    print(
        "    Y90_dose_3plane_isodose.png"
    )

    print(
        "    Y90_dose_3D_isosurfaces.png"
    )

    print(
        "    Y90_lung_segment_DVH.png"
    )

    print(
        "    Lu177_dose_Gy.npy"
    )

    print(
        "    right_lung_mask.npy / left_lung_mask.npy"
    )
    print(
        "    lung_geometry_reference_audit.csv"
    )

    print(
        "    Lu177_dose_3plane_isodose.png"
    )

    print(
        "    Lu177_dose_3D_isosurfaces.png"
    )

    print(
        "    Lu177_lung_segment_DVH.png"
    )

    print()
    print(
        "A DPK energy-conservation audit was performed for each isotope."
    )

    print(
        "MCDB S-values were NOT used."
    )

    print(
        "Dose was calculated using the existing Graves reference-density "
        "DPK/radiological-distance approximation, followed by voxel-wise "
        "CT-HU-derived density correction for local mass. This is not a "
        "full path-dependent heterogeneous electron transport calculation."
    )

    print()
    print(
        "The final SPECT, activity and dose arrays are "
        "all on the exact CT voxel grid."
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()