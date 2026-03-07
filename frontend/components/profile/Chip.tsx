import { Pressable, Text, View } from "react-native";
import { useTheme } from "@/constants/theme";

export function Chip({
  label,
  active,
  onPress,
  onRemove,
  color,
}: Readonly<{
  label: string;
  active?: boolean;
  onPress?: () => void;
  onRemove?: () => void;
  color?: string;
}>) {
  const t = useTheme();
  const c = color ?? t.accent;

  if (active !== undefined) {
    // Toggle chip
    return (
      <Pressable
        onPress={onPress}
        style={{
          flexDirection: "row",
          alignItems: "center",
          backgroundColor: active ? c + "20" : t.surface,
          borderRadius: 10,
          paddingHorizontal: 13,
          paddingVertical: 7,
          borderWidth: 1,
          borderColor: active ? c + "55" : t.border,
        }}
      >
        <Text
          style={{
            fontSize: 12,
            fontFamily: "Poppins_600SemiBold",
            color: active ? c : t.textMuted,
          }}
        >
          {label}
        </Text>
      </Pressable>
    );
  }

  // Tag chip (removable)
  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: c + "18",
        borderRadius: 10,
        paddingLeft: 11,
        paddingRight: onRemove ? 6 : 11,
        paddingVertical: 6,
        borderWidth: 1,
        borderColor: c + "40",
        gap: 4,
      }}
    >
      <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: c }}>
        {label}
      </Text>
      {onRemove && (
        <Pressable onPress={onRemove} hitSlop={10} style={{ padding: 2 }}>
          <Text style={{ fontSize: 14, lineHeight: 16, color: c + "99", fontFamily: "Poppins_600SemiBold" }}>
            ×
          </Text>
        </Pressable>
      )}
    </View>
  );
}