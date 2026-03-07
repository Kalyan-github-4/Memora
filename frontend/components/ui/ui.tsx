import { useTheme } from "@/constants/theme";
import type { ContactStatus } from "@/types";
import { Text, View } from "react-native";

interface AvatarProps {
  initials: string;
  color: string;
  size?: "sm" | "md" | "lg";
}

export function Avatar({ initials, color, size = "md" }: AvatarProps) {
  const dimensions = { sm: 36, md: 44, lg: 64 };
  const radii = { sm: 11, md: 13, lg: 20 };
  const fontSizes = { sm: 11, md: 13, lg: 20 };
  const dim = dimensions[size];

  return (
    <View
      style={{
        width: dim,
        height: dim,
        borderRadius: radii[size],
        backgroundColor: color + "1A",
        borderWidth: 1.5,
        borderColor: color + "55",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Text style={{ fontSize: fontSizes[size], fontFamily: "Syne_700Bold", color }}>
        {initials}
      </Text>
    </View>
  );
}

interface ScoreBarProps {
  score: number;
  color: string;
}

export function ScoreBar({ score, color }: ScoreBarProps) {
  const t = useTheme();
  return (
    <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
      <View style={{
        flex: 1,
        height: 3,
        backgroundColor: t.border,
        borderRadius: 2,
        overflow: "hidden",
      }}>
        <View
          style={{
            height: "100%",
            width: `${score}%`,
            backgroundColor: color,
            borderRadius: 2,
          }}
        />
      </View>
      <Text style={{
        fontSize: 11,
        fontFamily: "Poppins_600SemiBold",
        color: t.textMuted,
        width: 28,
        textAlign: "right",
      }}>
        {score}%
      </Text>
    </View>
  );
}

interface StatusDotProps {
  status: ContactStatus;
}

export function StatusDot({ status }: StatusDotProps) {
  const t = useTheme();
  const colors: Record<ContactStatus, string> = {
    hot: t.orange,
    warm: t.accent,
    cold: t.textMuted,
  };

  return (
    <View
      style={{
        width: 7,
        height: 7,
        borderRadius: 4,
        backgroundColor: colors[status],
      }}
    />
  );
}
