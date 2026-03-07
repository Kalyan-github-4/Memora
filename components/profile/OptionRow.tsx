import { Text, View } from "react-native";
import { useTheme } from "@/constants/theme";
import { Chip } from "@/components/profile/Chip";

export function OptionRow<T extends string>({
  label,
  options,
  value,
  onChange,
}: Readonly<{
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}>) {
  const t = useTheme();
  return (
    <View style={{ paddingHorizontal: 16, paddingVertical: 13 }}>
      <Text
        style={{
          fontSize: 10,
          fontFamily: "Poppins_600SemiBold",
          color: t.textSub,
          textTransform: "uppercase",
          letterSpacing: 1.1,
          marginBottom: 10,
        }}
      >
        {label}
      </Text>
      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
        {options.map((opt) => (
          <Chip
            key={opt.value}
            label={opt.label}
            active={value === opt.value}
            onPress={() => onChange(opt.value)}
          />
        ))}
      </View>
    </View>
  );
}