const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require('nativewind/metro');

const config = getDefaultConfig(__dirname);

// Allow bundling of .onnx model files as binary assets
config.resolver.assetExts.push('onnx');

module.exports = withNativeWind(config, { input: './global.css' });