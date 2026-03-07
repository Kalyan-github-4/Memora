import React from "react";
import { Text, View, TextProps, ViewProps, StyleSheet } from "react-native";
import { useTheme } from "@/constants/theme";

type ThemedTextProps = TextProps & {
  type?: "default" | "title" | "link";
};

export function ThemedView({ style, ...props }: ViewProps) {
  const t = useTheme();

  return (
    <View
      style={[
        {
          backgroundColor: t.bg,
        },
        style,
      ]}
      {...props}
    />
  );
}

export function ThemedText({ type = "default", style, ...props }: ThemedTextProps) {
  const t = useTheme();

  return (
    <Text
      style={[
        { color: t.text },
        type === "title" && styles.title,
        type === "link" && { color: t.accent },
        style,
      ]}
      {...props}
    />
  );
}

const styles = StyleSheet.create({
  title: {
    fontSize: 28,
    fontWeight: "700",
  },
});