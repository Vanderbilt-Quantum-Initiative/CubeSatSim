

def evaluate_point (
    geometry: Geometry,
    params: dict,
    source: SourceConfig,
    detector: DetectorModel,
    fading_model: FadingModel | None = None,
    cn2_profile: Cn2Profile | None = None,
) -> LinkState:
    
    """
    Full chain at a single geometry:
    1. link_loss → LossBudget
    2. atmosphere + turbulence → FadingModel (if not precomputed)
    3. For each intensity (μ, ν, vacuum):
         detection → DetectionResult (gain AND QBER integrated over fading)
    4. Assemble → LinkState

    Decoy bounds and key rate are NOT computed here — they require
    accumulated statistics across the full pass. The pass simulator
    calls this per-timestep, accumulates, then runs decoy + keyrate.
    """