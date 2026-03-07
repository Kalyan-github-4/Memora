import { SearchIcon } from "@/components/ui/icons";
import { mockContacts } from "@/constants/data";
import { useTheme } from "@/constants/theme";
import { contactScore } from "@/utils/contactScore";
import { contactStrength } from "@/utils/contactStrength";
import { router } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useMemo, useState } from "react";
import { Pressable, ScrollView, Text, TextInput, View } from "react-native";
import { Image } from "expo-image";
import defaultAvatar from "../../assets/images/default-image.jpeg"
import { LinearGradient } from "expo-linear-gradient";

export default function VaultScreen() {
  const t = useTheme();
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "hot" | "warm" | "cold">("all");

  const filtered = useMemo(() => {
    return mockContacts.filter((c) => {
      const q = search.toLowerCase();
      const matchSearch =
        !q ||
        c.name.toLowerCase().includes(q) ||
        c.role.toLowerCase().includes(q) ||
        c.company.toLowerCase().includes(q) ||
        c.topics.some((topic) => topic.toLowerCase().includes(q));
      const matchFilter = filter === "all" || contactStrength(c) === filter;
      return matchSearch && matchFilter;
    });
  }, [search, filter]);

  const filters = ["all", "hot", "warm", "cold"] as const;

  const filterColors: Record<string, string> = {
    hot: t.orange,
    warm: t.accent,
    cold: t.textMuted,
    all: t.accent,
  };

  return (
    <View style={{ flex: 1, backgroundColor: t.bg }}>
      <StatusBar style="auto" />

      {/* Top gradient glow */}
      <LinearGradient
        colors={[t.accent + "28", t.purple + "12", "transparent"]}
        locations={[0, 0.5, 1]}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 320,
          pointerEvents: "none",
        }}
      />
      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={{ paddingBottom: 120, paddingHorizontal: 20 }}
        showsVerticalScrollIndicator={false}
      >

        {/* Header */}
        <View style={{ paddingTop: 56, paddingBottom: 24 }}>
          <Text style={{ fontSize: 26, fontFamily: "Syne_800ExtraBold", color: t.text }}>
            Memory Vault
          </Text>
          <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.textMuted, marginTop: 4 }}>
            {mockContacts.length} connections captured
          </Text>
        </View>

        {/* Search */}
        <View style={{
          flexDirection: "row",
          alignItems: "center",
          gap: 10,
          backgroundColor: t.card,
          borderWidth: 1,
          borderColor: t.border,
          borderRadius: 14,
          paddingVertical: 12,
          paddingHorizontal: 14,
          marginBottom: 14,
        }}>
          <SearchIcon size={16} color={t.textMuted} />
          <TextInput
            value={search}
            onChangeText={setSearch}
            placeholder='Try "ML guy from Mumbai"...'
            placeholderTextColor={t.textMuted}
            style={{
              flex: 1,
              fontSize: 13,
              fontFamily: "Poppins_400Regular",
              color: t.text,
              padding: 0,
            }}
          />
        </View>

        {/* Filters */}
        <View style={{ flexDirection: "row", gap: 8, marginBottom: 24 }}>
          {filters.map((f) => {
            const isActive = filter === f;

            const pillContent = (
              <Pressable
                onPress={() => setFilter(f)}
                style={{
                  paddingVertical: 6,
                  paddingHorizontal: 16,
                  borderRadius: 20,
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Text
                  style={{
                    fontSize: 12,
                    fontFamily: "Poppins_600SemiBold",
                    textTransform: "capitalize",
                    color: isActive ? "#fff" : t.textMuted,
                  }}
                >
                  {f}
                </Text>
              </Pressable>
            );

            if (isActive) {
              return (
                <LinearGradient
                  key={f}
                  colors={[t.accent, t.purple]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={{
                    borderRadius: 20,
                  }}
                >
                  {pillContent}
                </LinearGradient>
              );
            }

            return (
              <View
                key={f}
                style={{
                  backgroundColor: t.card,
                  borderWidth: 1,
                  borderColor: t.border,
                  borderRadius: 20,
                }}
              >
                {pillContent}
              </View>
            );
          })}
        </View>

        {/* Contacts List */}
        <View style={{ gap: 12 }}>
          {filtered.map((c) => {
            const { score, color } = contactScore(c);
            return (
              <Pressable
                key={c.id}
                onPress={() => router.push({ pathname: "/contact", params: { id: String(c.id) } })}
                style={({ pressed }) => ({
                  backgroundColor: pressed ? t.cardHighlight : t.card,
                  borderRadius: 20,
                  borderWidth: 1,
                  borderColor: t.border,
                  padding: 16,
                })}
              >
                {/* Top Row */}
                <View style={{ flexDirection: "row", alignItems: "center", gap: 12, marginBottom: 14 }}>
                  <View style={{
                    width: 46,
                    height: 46,
                    borderRadius: 14,
                    backgroundColor: c.color + "1A",
                    borderWidth: 1.5,
                    borderColor: c.color + "50",
                    alignItems: "center",
                    justifyContent: "center",
                  }}>
                    <Image source={c.img ? { uri: c.img } : defaultAvatar} style={{ width: 46, height: 46, borderRadius: 14 }} />
                    {/* <Text style={{ fontSize: 14, fontFamily: "Syne_700Bold", color: c.color }}>
                      {c.avatar}
                    </Text> */}
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={{ fontSize: 15, fontFamily: "Syne_700Bold", color: t.text }}>
                      {c.name}
                    </Text>
                    <Text style={{ fontSize: 12, fontFamily: "Poppins_400Regular", color: t.textMuted, marginTop: 2 }}>
                      {c.role} · {c.company}
                    </Text>
                  </View>
                  <Text style={{ fontSize: 13, fontFamily: "Poppins_600SemiBold", color: color }}>
                    {score}%
                  </Text>
                </View>

                {/* Score bar */}
                <View style={{
                  height: 3,
                  backgroundColor: t.border,
                  borderRadius: 2,
                  marginBottom: 12,
                  overflow: "hidden",
                }}>
                  <View style={{ height: "100%", width: `${score}%`, backgroundColor: color, borderRadius: 2 }} />
                </View>

                {/* Topics */}
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
                  {c.topics.slice(0, 3).map((topic) => (
                    <View
                      key={topic}
                      style={{
                        backgroundColor: t.surface,
                        borderWidth: 1,
                        borderColor: t.border,
                        borderRadius: 8,
                        paddingVertical: 3,
                        paddingHorizontal: 10,
                      }}
                    >
                      <Text style={{ fontSize: 11, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
                        {topic}
                      </Text>
                    </View>
                  ))}
                </View>

                {/* Met At */}
                <Text style={{ fontSize: 11, fontFamily: "Poppins_400Regular", color: t.textSub }}>
                  {c.metAt}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </ScrollView>
    </View>
  );
}
