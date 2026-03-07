import { Text } from "react-native";
import { useTheme } from "@/constants/theme"


export function SectionLabel({ title }: Readonly<{ title: string }>) {
  const t = useTheme();
  return (
    <Text
      style={{
        fontSize: 11,
        fontFamily: "Poppins_600SemiBold",
        color: t.textMuted,
        textTransform: "uppercase",
        letterSpacing: 1.4,
        marginBottom: 10,
        marginLeft: 2,
      }}
    >
      {title}
    </Text>
  );
}