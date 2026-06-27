# Cibuco_Boriken — CFAR Phase 2: Clutter Suppression
# Reviewed by: Claude (Sonnet 4.6)
# Architect: Danny (Cibuco_Boriken) 🇵🇷
# Date: March 12, 2026
#
# Radar concept: MTI (Moving Target Indicator)
# Bioacoustic translation: Ambient soundscape clutter suppression
#
# Pass to Opus with:
# "Implement CFAR Phase 2 clutter suppression.
#  Add to existing cfar_threshold.py and evaluate_thresholds.py.
#  Do not break existing Phase 1 functionality."

# ══════════════════════════════════════════════════════
# PHASE 2A — CLUTTER SUPPRESSION MODULE
# Add to: birdclef/cfar_threshold.py
# ══════════════════════════════════════════════════════

PHASE_2A = [

    {
        "id": "2A-01",
        "task": "Add estimate_clutter_profile() to cfar_threshold.py",
        "radar_analogy": "Clutter map generation — radar builds map of static returns before scan",
        "details": """
        def estimate_clutter_profile(
            all_window_probs: np.ndarray,
            percentile: float = 25.0,
        ) -> np.ndarray:
            '''
            Estimate ambient clutter profile for a soundscape.
            
            Analogous to radar clutter map:
              - Takes ALL windows from one soundscape
              - Computes per-species baseline activation level
              - This baseline = ambient clutter level
              - Species genuinely present will be ABOVE this
              
            Args:
                all_window_probs: shape (num_windows, num_species)
                                  ALL windows from one soundscape
                percentile: float, default 25.0
                            Lower percentile = more aggressive suppression
                            25th percentile captures ambient without
                            suppressing genuine detections
                            
            Returns:
                clutter_map: shape (num_species,)
                             per-species ambient baseline
                             
            Pantanal examples:
                Dawn chorus → all species elevated → high clutter map
                              CFAR will see above-chorus detections only
                Rain event  → uniform elevation → captured in clutter map
                              species above rain level still detected
                Quiet night → near-zero clutter → CFAR very sensitive
            '''
            # Use 25th percentile as ambient estimate
            # NOT mean (biased by strong detections)
            # NOT median (too aggressive for sparse species)
            clutter_map = np.percentile(all_window_probs, 
                                        percentile, axis=0)
            return clutter_map.astype(np.float32)
        """,
    },

    {
        "id": "2A-02",
        "task": "Add suppress_clutter() to cfar_threshold.py",
        "radar_analogy": "MTI subtraction — subtract static clutter map from each new scan",
        "details": """
        def suppress_clutter(
            probs: np.ndarray,
            clutter_map: np.ndarray,
            floor: float = 0.0,
        ) -> np.ndarray:
            '''
            Subtract clutter map from activation probabilities.
            Analogous to MTI clutter cancellation in radar.
            
            After subtraction:
              - Species at ambient level → near zero (suppressed)
              - Species ABOVE ambient → positive residual (detected)
              - Rain/wind/chorus effects → removed uniformly
              
            Args:
                probs: shape (num_windows, num_species) OR (num_species,)
                       raw sigmoid probabilities
                clutter_map: shape (num_species,)
                             output of estimate_clutter_profile()
                floor: float, default 0.0
                       clip floor after subtraction
                       prevents negative probabilities
                       
            Returns:
                suppressed: same shape as probs
                            clutter-cancelled activations
                            
            Note: Apply BEFORE cfar_adaptive_threshold()
                  Pipeline: raw_probs
                            → suppress_clutter()    (MTI)
                            → cfar_adaptive_threshold() (CFAR)
                            → apply_threshold()     (detection)
            '''
            suppressed = probs - clutter_map
            suppressed = np.clip(suppressed, floor, 1.0)
            return suppressed.astype(np.float32)
        """,
    },

    {
        "id": "2A-03",
        "task": "Add combined pipeline function to cfar_threshold.py",
        "radar_analogy": "Full MTI-CFAR processing chain",
        "details": """
        def cfar_with_clutter_suppression(
            all_window_probs: np.ndarray,
            k: float = 2.0,
            clutter_percentile: float = 25.0,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            '''
            Full radar-inspired detection pipeline:
              Step 1: Estimate clutter map (MTI map generation)
              Step 2: Subtract clutter (MTI cancellation)  
              Step 3: CFAR threshold on clean signal
              Step 4: Binary detection decision
              
            Args:
                all_window_probs: (num_windows, num_species)
                k: CFAR sensitivity constant
                clutter_percentile: ambient estimation percentile
                
            Returns:
                detections: (num_windows, num_species) binary
                thresholds: (num_species,) per-species thresholds
                clutter_map: (num_species,) estimated clutter
                
            This is the FULL CONTRIBUTION of the paper.
            Three comparison conditions:
              Baseline:   fixed t=0.5
              Phase 1:    CFAR only (no clutter suppression)
              Phase 2:    CFAR + clutter suppression (MTI-CFAR)
            '''
            # Step 1: Build clutter map
            clutter_map = estimate_clutter_profile(
                all_window_probs, clutter_percentile
            )
            
            # Step 2: MTI suppression
            clean_probs = suppress_clutter(all_window_probs, clutter_map)
            
            # Step 3: CFAR on clean signal
            thresholds = cfar_adaptive_threshold(clean_probs, k=k)
            
            # Step 4: Detection
            detections = apply_threshold(clean_probs, thresholds)
            
            return detections, thresholds, clutter_map
        ''',
    },

]


# ══════════════════════════════════════════════════════
# PHASE 2B — EVALUATION UPDATE
# Update: birdclef/evaluate_thresholds.py
# ══════════════════════════════════════════════════════

PHASE_2B = [

    {
        "id": "2B-01",
        "task": "Add third comparison condition to k-sweep table",
        "details": """
        Update evaluate_thresholds.py to report THREE conditions:
        
        Condition 1: Fixed threshold (t=0.5)
        Condition 2: CFAR only (Phase 1, existing)
        Condition 3: CFAR + Clutter Suppression (Phase 2, new)
        
        New table format:
        
        k    | F1_fixed | F1_cfar | F1_mti_cfar | FPR_fixed | FPR_cfar | FPR_mti_cfar | AUC
        -----|----------|---------|-------------|-----------|----------|--------------|-----
        1.0  | 0.7670   | 0.6893  | ???         | 0.0002    | 0.0135   | ???          | ???
        
        Also add:
        - clutter_mean: average clutter map value across species
          (shows how much ambient noise was suppressed)
        - delta_f1_mti: F1_mti_cfar - F1_fixed
          (our key result — does full pipeline beat baseline?)
        """,
    },

    {
        "id": "2B-02",
        "task": "Add clutter profile diagnostics to output",
        "details": """
        After each k evaluation print:
        
        Clutter Profile:
          mean:  X.XXXX  (average ambient level)
          max:   X.XXXX  (most cluttered species)
          min:   X.XXXX  (cleanest species)
          
        Top 3 most cluttered species: [sp1, sp2, sp3]
        Top 3 cleanest species: [sp4, sp5, sp6]
        
        This is scientifically interesting:
        - Most cluttered = common background species
        - Cleanest = rare species (low baseline activation)
        - Tells us which species BENEFIT most from suppression
        """,
    },

    {
        "id": "2B-03",
        "task": "Update k_sweep_figure.png to show all 3 conditions",
        "details": """
        Update figure to show:
        
        Plot 1 (left): F1 vs k
          Line 1: Fixed t=0.5 (horizontal, dashed gray)
          Line 2: CFAR only (blue)
          Line 3: MTI-CFAR (green) ← new
          
        Plot 2 (right): FPR vs k  
          Line 1: Fixed t=0.5 (horizontal, dashed gray)
          Line 2: CFAR only (red)
          Line 3: MTI-CFAR (orange) ← new
          
        This becomes Figure 1 in the paper.
        Shows the full radar pipeline advantage visually.
        """,
    },

]


# ══════════════════════════════════════════════════════
# PHASE 2C — COLAB RUN SEQUENCE
# After Opus pushes to GitHub
# ══════════════════════════════════════════════════════

COLAB_RUN_SEQUENCE = [

    {
        "id": "C-01",
        "cell": "Pull latest code",
        "command": "!git pull origin main",
    },

    {
        "id": "C-02",
        "cell": "Run full dataset training on T4 GPU",
        "command": """
            # Switch to T4 GPU first: Runtime → Change runtime type → T4
            !BIRDCLEF_DATA_DIR=/content/cibuco-boriken/data/birdclef-2026 \\
              python -m birdclef.train \\
              --backbone small \\
              --epochs 20 \\
              --include-soundscapes
        """,
        "expected_time": "~45-60 min on T4",
        "note": "Full 35K samples. This is the paper training run.",
    },

    {
        "id": "C-03",
        "cell": "Run Phase 2 k-sweep with clutter suppression",
        "command": """
            !BIRDCLEF_DATA_DIR=/content/cibuco-boriken/data/birdclef-2026 \\
              python -m birdclef.evaluate_thresholds \\
              --backbone small \\
              --include-soundscapes \\
              --k-sweep 1.0 1.5 2.0 2.5 3.0
        """,
        "watching_for": [
            "F1_mti_cfar > F1_fixed at some k  ← PAPER RESULT",
            "Clutter mean >> 0 (real suppression happening)",
            "Visible knee in 3-line figure",
            "Top cluttered species = common waterbirds",
            "Top clean species = rare mammals/reptiles",
        ],
    },

]


# ══════════════════════════════════════════════════════
# PAPER CONTRIBUTION SUMMARY
# ══════════════════════════════════════════════════════

PAPER_CONTRIBUTIONS = {
    "Title": (
        "From Radar to Rainforest: CFAR-Inspired Adaptive Detection "
        "with Clutter Suppression for Biodiversity Monitoring "
        "in the Brazilian Pantanal"
    ),
    "Team": "Cibuco_Boriken 🇵🇷🌿",
    "Contribution_1": (
        "Per-species CFAR adaptive thresholding — "
        "analogous to per-target CFAR in AESA radar systems"
    ),
    "Contribution_2": (
        "Soundscape clutter suppression via ambient profile estimation — "
        "analogous to MTI clutter cancellation"
    ),
    "Contribution_3": (
        "Minimum data threshold for CFAR effectiveness — "
        "n≥20 samples/species required for non-degenerate noise floor"
    ),
    "Novelty": "First application of radar signal processing pipeline to bioacoustic species detection",
    "Originality_score": "5/5 — Trailblazing",
}
