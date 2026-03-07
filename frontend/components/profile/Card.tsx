import { View } from "react-native";
import { useTheme } from "@/constants/theme"

export function Card({ children }: Readonly<{ children: React.ReactNode }>) {
  const t = useTheme();
  return (
    <View
      style={{
        backgroundColor: t.card,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: t.border,
        overflow: "hidden",
        marginBottom: 20,
      }}
    >
      {children}
    </View>
  );
}
