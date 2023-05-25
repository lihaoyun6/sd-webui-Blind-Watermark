import launch

if not launch.is_installed("pywt"):
    launch.run_pip("install pywavelets", "pywavelets for BlindWatermark")

if not launch.is_installed("base58"):
    launch.run_pip("install base58", "base58 for BlindWatermark")