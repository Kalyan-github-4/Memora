import { View } from "react-native";
import { useTheme } from "@/constants/theme";
export function Divider() {
  const t = useTheme();
  return <View style={{ height: 1, backgroundColor: t.border, marginLeft: 16 }} />;
}

// ─── Field Row ────────────────────────────────────────────────────────────────

// function FieldRow({
//   label,
//   value,
//   onChange,
//   placeholder,
//   multiline,
// }: Readonly<{
//   label: string;
//   value: string;
//   onChange: (v: string) => void;
//   placeholder?: string;
//   multiline?: boolean;
// }>) {
//   const t = useTheme();
//   return (
//     <View style={{ paddingHorizontal: 16, paddingVertical: 13 }}>
//       <Text
//         style={{
//           fontSize: 10,
//           fontFamily: "Poppins_600SemiBold",
//           color: t.textSub,
//           textTransform: "uppercase",
//           letterSpacing: 1.1,
//           marginBottom: 5,
//         }}
//       >
//         {label}
//       </Text>
//       <TextInput
//         value={value}
//         onChangeText={onChange}
//         placeholder={placeholder ?? label}
//         placeholderTextColor={t.textSub}
//         multiline={multiline}
//         style={{
//           fontSize: 15,
//           fontFamily: "Poppins_400Regular",
//           color: t.text,
//           lineHeight: multiline ? 22 : 20,
//           minHeight: multiline ? 68 : undefined,
//           textAlignVertical: multiline ? "top" : "center",
//           paddingTop: multiline ? 2 : 0,
//         }}
//       />
//     </View>
//   );
// }
