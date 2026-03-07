import { Pressable, Text, TextInput, View } from "react-native";
import { useTheme } from "@/constants/theme";
import { useRef, useState } from "react";
import { Chip } from "@/components/profile/Chip";

export function TagRow({
  label,
  tags,
  onAdd,
  onRemove,
  placeholder,
  color,
}: Readonly<{
  label: string;
  tags: string[];
  onAdd: (tag: string) => void;
  onRemove: (tag: string) => void;
  placeholder?: string;
  color?: string;
}>) {
  const t = useTheme();
  const [editing, setEditing] = useState(false);
  const [input, setInput] = useState("");
  const inputRef = useRef<TextInput>(null);

  const commit = () => {
    const v = input.trim();
    if (v && !tags.map((x) => x.toLowerCase()).includes(v.toLowerCase())) {
      onAdd(v);
    }
    setInput("");
    setEditing(false);
  };

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
        {tags.map((tag) => (
          <Chip
            key={tag}
            label={tag}
            onRemove={() => onRemove(tag)}
            color={color}
          />
        ))}

        {editing ? (
          <TextInput
            ref={inputRef}
            autoFocus
            value={input}
            onChangeText={setInput}
            onSubmitEditing={commit}
            onBlur={commit}
            placeholder={placeholder ?? "Type + enter"}
            placeholderTextColor={t.textSub}
            returnKeyType="done"
            style={{
              fontSize: 12,
              fontFamily: "Poppins_400Regular",
              color: t.text,
              backgroundColor: t.surface,
              borderWidth: 1,
              borderColor: t.accent + "70",
              borderRadius: 10,
              paddingHorizontal: 11,
              paddingVertical: 6,
              minWidth: 120,
            }}
          />
        ) : (
          <Pressable
            onPress={() => setEditing(true)}
            style={{
              flexDirection: "row",
              alignItems: "center",
              backgroundColor: t.surface,
              borderRadius: 10,
              paddingHorizontal: 11,
              paddingVertical: 6,
              borderWidth: 1,
              borderColor: t.border,
              borderStyle: "dashed",
            }}
          >
            <Text
              style={{
                fontSize: 12,
                fontFamily: "Poppins_600SemiBold",
                color: t.textMuted,
              }}
            >
              + Add
            </Text>
          </Pressable>
        )}
      </View>
    </View>
  );
}