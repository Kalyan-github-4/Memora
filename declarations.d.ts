declare module "*.png";
declare module "*.jpg";
declare module "*.jpeg";
declare module "*.onnx";

// react-native-audio-record does not ship its own types
declare module "react-native-audio-record" {
  interface AudioRecordOptions {
    sampleRate: number;
    channels: number;
    bitsPerSample: number;
    /** Filename written inside the OS cache dir (used by the native layer). */
    wavFile: string;
  }

  const AudioRecord: {
    init(options: AudioRecordOptions): void;
    start(): void;
    stop(): Promise<string>;
    on(event: "data", callback: (base64Chunk: string) => void): void;
    removeEventListener(event: "data"): void;
  };

  export default AudioRecord;
}
