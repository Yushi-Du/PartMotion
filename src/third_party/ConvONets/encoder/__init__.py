from src.third_party.ConvONets.encoder import encoder

encoder_dict = {
    "pointnetpp": encoder.LocalPoolPointnetPPFusion,
    "pointnetpp_4dims": encoder.LocalPoolPointnetPPFusion_4dims,
    "pointnetpp_4dims_3frames": encoder.LocalPoolPointnetPPFusion_4dims_3frame,
    "pointnetpp_4dims_3frames_interpolation": encoder.LocalPoolPointnetPPFusion_4dims_3frame_interpolation,
    "pointnetpp_4dims_3frames_extrapolation": encoder.LocalPoolPointnetPPFusion_4dims_3frame_extrapolation,
    "pointnetpp_4dims_implicit": encoder.LocalPoolPointnetPPFusion_4dims_GlobalCode,
    "ablation": encoder.OnlyPointnet_4dims_GlobalCode,
}
