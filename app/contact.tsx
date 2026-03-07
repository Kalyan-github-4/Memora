import { mockContacts } from "@/constants/data";
import { useTheme } from "@/constants/theme";
import { contactScore } from "@/utils/contactScore";
import { contactStrength } from "@/utils/contactStrength";
import { useLocalSearchParams } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useState } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { Image } from "expo-image";
import defaultAvatar from "../assets/images/default-image.jpeg"

const strengthConfig = {
  hot: { label: "Hot", color: "#FB923C" },
  warm: { label: "Warm", color: "#7C6FFF" },
  cold: { label: "Cold", color: "#7878A0" },
};

export default function ContactScreen() {
  const t = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const c = mockContacts.find((m) => m.id === Number(id)) ?? mockContacts[0];
  const [copied, setCopied] = useState(false);

  const { score, color } = contactScore(c);
  const strength = contactStrength(c);
  const { label: strengthLabel, color: strengthColor } = strengthConfig[strength];

  const handleCopy = () => {
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <View style={{ flex: 1, backgroundColor: t.bg }}>
      <StatusBar style="auto" />
      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={{ paddingBottom: 48 }}
        showsVerticalScrollIndicator={false}
      >
        {/* Hero */}
        <View
          style={{
            paddingTop: 24,
            paddingBottom: 28,
            paddingHorizontal: 24,
            borderBottomWidth: 1,
            borderBottomColor: t.border,
            backgroundColor: c.color + "0C",
          }}
        >
          <View style={{ flexDirection: "row", alignItems: "center", gap: 16, marginBottom: 20 }}>
            <View style={{
              width: 64,
              height: 64,
              borderRadius: 20,
              backgroundColor: c.color + "20",
              borderWidth: 2,
              borderColor: c.color + "60",
              alignItems: "center",
              justifyContent: "center",
            }}>
              <Image source={c.img ? { uri: c.img } : defaultAvatar} style={{ width: 64, height: 64, borderRadius: 20 }} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={{ fontSize: 22, fontFamily: "Syne_800ExtraBold", color: t.text, lineHeight: 28 }}>
                {c.name}
              </Text>
              <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.textMuted, marginTop: 2 }}>
                {c.role} · {c.company}
              </Text>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginTop: 8 }}>
                <View style={{
                  paddingHorizontal: 10,
                  paddingVertical: 3,
                  borderRadius: 8,
                  backgroundColor: strengthColor + "18",
                  borderWidth: 1,
                  borderColor: strengthColor + "50",
                }}>
                  <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: strengthColor }}>
                    {strengthLabel}
                  </Text>
                </View>
                <Text style={{ fontSize: 12, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
                  {c.location}
                </Text>
              </View>
            </View>
          </View>

          {/* Score Bar */}
          <View>
            <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
              <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1 }}>
                Connection Strength
              </Text>
              <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: color }}>
                {score}%
              </Text>
            </View>
            <View style={{ height: 4, backgroundColor: t.border, borderRadius: 2, overflow: "hidden" }}>
              <View style={{ height: "100%", width: `${score}%`, backgroundColor: color, borderRadius: 2 }} />
            </View>
          </View>
        </View>

        <View style={{ paddingHorizontal: 24, paddingTop: 24, gap: 20 }}>
          {/* Met At */}
          <View style={{
            backgroundColor: t.card,
            borderRadius: 18,
            borderWidth: 1,
            borderColor: t.border,
            padding: 16,
          }}>
            <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>
              Met At
            </Text>
            <Text style={{ fontSize: 14, fontFamily: "Poppins_600SemiBold", color: t.text }}>
              {c.metAt}
            </Text>
            <Text style={{ fontSize: 12, fontFamily: "Poppins_400Regular", color: t.textMuted, marginTop: 4 }}>
              {c.metDate}
            </Text>
          </View>

          {/* Summary */}
          <View style={{
            backgroundColor: t.card,
            borderRadius: 18,
            borderWidth: 1,
            borderColor: t.border,
            padding: 16,
          }}>
            <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>
              Conversation Summary
            </Text>
            <Text style={{ fontSize: 14, fontFamily: "Poppins_400Regular", color: t.text, lineHeight: 22 }}>
              {c.summary}
            </Text>
          </View>

          {/* Topics */}
          <View>
            <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
              Topics Discussed
            </Text>
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
              {c.topics.map((topic) => (
                <View
                  key={topic}
                  style={{
                    paddingVertical: 6,
                    paddingHorizontal: 14,
                    borderRadius: 20,
                    backgroundColor: c.color + "14",
                    borderWidth: 1,
                    borderColor: c.color + "40",
                  }}
                >
                  <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: c.color }}>
                    {topic}
                  </Text>
                </View>
              ))}
            </View>
          </View>

          {/* Follow-ups */}
          <View>
            <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
              Follow-Up Actions
            </Text>
            <View style={{ gap: 8 }}>
              {c.followUps.map((f, i) => (
                <View
                  key={i}
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 12,
                    backgroundColor: t.card,
                    borderWidth: 1,
                    borderColor: t.border,
                    borderRadius: 14,
                    paddingVertical: 12,
                    paddingHorizontal: 14,
                  }}
                >
                  <View style={{
                    width: 20,
                    height: 20,
                    borderRadius: 6,
                    borderWidth: 2,
                    borderColor: t.orange,
                  }} />
                  <Text style={{ flex: 1, fontSize: 13, fontFamily: "Poppins_400Regular", color: t.text }}>
                    {f}
                  </Text>
                </View>
              ))}
            </View>
          </View>

          {/* AI Follow-up Draft */}
          <View style={{
            padding: 18,
            borderRadius: 18,
            backgroundColor: t.accentDim,
            borderWidth: 1,
            borderColor: t.accent + "30",
          }}>
            <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.accent, textTransform: "uppercase", letterSpacing: 1 }}>
                AI Follow-Up Draft
              </Text>
              <Pressable onPress={handleCopy}>
                <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: copied ? t.green : t.accent }}>
                  {copied ? "Copied!" : "Copy"}
                </Text>
              </Pressable>
            </View>
            <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.text, lineHeight: 22, fontStyle: "italic" }}>
              &quot;Hey {c.name.split(" ")[0]}, really enjoyed our conversation at {c.metAt}. {c.followUps[0]} as promised. Would love to stay in touch!&quot;
            </Text>
          </View>

          {/* Contact Info */}
          {c.contactInfo.length > 0 && (
            <View>
              <Text style={{ fontSize: 11, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
                Contact Info
              </Text>
              <View style={{
                backgroundColor: t.card,
                borderRadius: 18,
                borderWidth: 1,
                borderColor: t.border,
                overflow: "hidden",
              }}>
                {c.contactInfo.map((info, i) => (
                  <View
                    key={i}
                    style={{
                      paddingVertical: 12,
                      paddingHorizontal: 16,
                      borderTopWidth: i > 0 ? 1 : 0,
                      borderTopColor: t.border,
                    }}
                  >
                    <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.text }}>
                      {info}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          )}
        </View>
      </ScrollView>
    </View>
  );
}
