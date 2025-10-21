from picamera2 import Picamera2

picam2 = Picamera2()
# picam2.start()  # démarre la caméra
print("Sensor_modes:", picam2.sensor_modes)        # Liste les modes du capteur
print("Camera_controls:", picam2.camera_controls)     # Paramètres actuels contrôlés par Picamera2

config = picam2.camera_configuration
print("Config:", config)

''' A renvoyé :
Sensor_modes: [{'format': SRGGB10_CSI2P, 'unpacked': 'SRGGB10', 'bit_depth': 10, 'size': (1536, 864), 'fps': 120.13, 'crop_limits': (768, 432, 3072, 1728), 'exposure_limits': (9, 77208145, 20000)}, {'format': SRGGB10_CSI2P, 'unpacked': 'SRGGB10', 'bit_depth': 10, 'size': (2304, 1296), 'fps': 56.03, 'crop_limits': (0, 0, 4608, 2592), 'exposure_limits': (13, 112015096, 20000)}, {'format': SRGGB10_CSI2P, 'unpacked': 'SRGGB10', 'bit_depth': 10, 'size': (4608, 2592), 'fps': 14.35, 'crop_limits': (0, 0, 4608, 2592), 'exposure_limits': (26, 220416802, 20000)}]
Camera_controls: {'ScalerCrop': ((0, 0, 64, 64), (0, 0, 4608, 2592), (576, 0, 3456, 2592)), 'NoiseReductionMode': (0, 4, 0), 'Sharpness': (0.0, 16.0, 1.0), 'ExposureValue': (-8.0, 8.0, 0.0), 'CnnEnableInputTensor': (False, True, False), 'AeFlickerMode': (0, 1, 0), 'ExposureTime': (26, 220416802, 20000), 'AeExposureMode': (0, 3, 0), 'ExposureTimeMode': (0, 1, 0), 'AeConstraintMode': (0, 3, 0), 'AeEnable': (False, True, True), 'StatsOutputEnable': (False, True, False), 'ColourCorrectionMatrix': (0.0, 8.0, None), 'AnalogueGain': (1.1228070259094238, 16.0, 1.0), 'SyncFrames': (1, 1000000, 100), 'AfMode': (0, 2, 0), 'Saturation': (0.0, 32.0, 1.0), 'ColourTemperature': (100, 100000, None), 'LensPosition': (0.0, 35.0, 1.0), 'AfTrigger': (0, 1, 0), 'AwbMode': (0, 7, 0), 'ColourGains': (0.0, 32.0, None), 'AfWindows': ((0, 0, 0, 0), (65535, 65535, 65535, 65535), (0, 0, 0, 0)), 'AwbEnable': (False, True, None), 'Contrast': (0.0, 32.0, 1.0), 'AeFlickerPeriod': (100, 1000000, None), 'AfSpeed': (0, 1, 0), 'AfMetering': (0, 1, 0), 'AeMeteringMode': (0, 3, 0), 'AfRange': (0, 2, 0), 'FrameDurationLimits': (69669, 220535845, 33333), 'AnalogueGainMode': (0, 1, 0), 'AfPause': (0, 2, 0), 'HdrMode': (0, 4, 0), 'SyncMode': (0, 2, 0), 'Brightness': (-1.0, 1.0, 0.0)}
Config: <bound method Picamera2.camera_configuration of <picamera2.picamera2.Picamera2 object at 0x7f8e4b5590>>'''