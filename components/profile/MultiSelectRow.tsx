import { Text, View } from "react-native";
import { useTheme } from "@/constants/theme";
import { Chip } from "@/components/profile/Chip";

export function MultiSelectRow<T extends string>({
  label,
  options,
  values,
  onChange,
  color,
}: Readonly<{
  label: string;
  options: { value: T; label: string }[];
  values: T[];
  onChange: (values: T[]) => void;
  color?: string;
}>) {
  const t = useTheme();
  const toggle = (v: T) =>
    onChange(values.includes(v) ? values.filter((x) => x !== v) : [...values, v]);

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
            active={values.includes(opt.value)}
            onPress={() => toggle(opt.value)}
            color={color ?? undefined}
          />
        ))}
      </View>
    </View>
  );
}