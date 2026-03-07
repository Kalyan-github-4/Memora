import { View, Text } from "react-native";
import { useTheme } from "@/constants/theme"


export function ProfileCompletion({ percent }: { percent: number }) {
  const t = useTheme();

  return (
    <View
      style={{
        backgroundColor: t.card,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: t.border,
        padding: 18,
        marginBottom: 28,
      }}
    >
      <Text
        style={{
          fontSize: 14,
          fontFamily: "Syne_700Bold",
          color: t.text,
          marginBottom: 10,
        }}
      >
        Complete your profile
      </Text>

      {/* Progress bar */}
      <View
        style={{
          height: 10,
          backgroundColor: t.surface,
          borderRadius: 20,
          overflow: "hidden",
        }}
      >
        <View
          style={{
            height: "100%",
            width: `${percent}%`,
            backgroundColor: t.accent,
          }}
        />
      </View>

      <Text
        style={{
          fontSize: 11,
          fontFamily: "Poppins_500Medium",
          color: t.textMuted,
          marginTop: 6,
        }}
      >
        {percent}% completed
      </Text>
    </View>
  );
}