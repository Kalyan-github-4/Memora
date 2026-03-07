import Svg, { Circle, Line, Path, Rect } from "react-native-svg";

interface IconProps {
  size?: number;
  color?: string;
}

export function MicIcon({ size = 24, color = "white" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Rect x={9} y={2} width={6} height={11} rx={3} fill={color} />
      <Path
        d="M5 10a7 7 0 0014 0"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
      />
      <Line
        x1={12} y1={17} x2={12} y2={21}
        stroke={color} strokeWidth={2} strokeLinecap="round"
      />
      <Line
        x1={9} y1={21} x2={15} y2={21}
        stroke={color} strokeWidth={2} strokeLinecap="round"
      />
    </Svg>
  );
}

export function HomeIcon({ size = 18, color = "#5A7A8A" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"
        stroke={color}
        strokeWidth={2}
      />
    </Svg>
  );
}

export function GridIcon({ size = 18, color = "#5A7A8A" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Rect x={3} y={3} width={7} height={7} rx={2} stroke={color} strokeWidth={2} />
      <Rect x={14} y={3} width={7} height={7} rx={2} stroke={color} strokeWidth={2} />
      <Rect x={3} y={14} width={7} height={7} rx={2} stroke={color} strokeWidth={2} />
      <Rect x={14} y={14} width={7} height={7} rx={2} stroke={color} strokeWidth={2} />
    </Svg>
  );
}

export function SettingsIcon({ size = 18, color = "#8BAAB8" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Circle cx={12} cy={12} r={3} stroke={color} strokeWidth={2} />
      <Path
        d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
      />
    </Svg>
  );
}

export function SearchIcon({ size = 16, color = "#5A7A8A" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Circle cx={11} cy={11} r={8} stroke={color} strokeWidth={2} />
      <Path
        d="m21 21-4.35-4.35"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
      />
    </Svg>
  );
}

export function BackIcon({ size = 18, color = "#00D4FF" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M19 12H5M12 19l-7-7 7-7"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function CheckIcon({ size = 12, color = "#050A0E" }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 12 12" fill="none">
      <Path
        d="M2 6l3 3 5-5"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}
