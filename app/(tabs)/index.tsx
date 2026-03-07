import { MicIcon } from "@/components/ui/icons";
import { mockContacts } from "@/constants/data";
import { useTheme } from "@/constants/theme";
import { useAuth } from "@/services/authContext";
import { contactScore } from "@/utils/contactScore";
import { router } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { Pressable, ScrollView, Text, View } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { getGreeting } from "@/utils/getGreeting";
import { Image } from "expo-image";
import defaultAvatar from "../../assets/images/default-image.jpeg"
import kalyanImg from "../../assets/images/kalyan.jpg"

export default function HomeScreen() {
  const t = useTheme();
  const { user } = useAuth();
  const recentContacts = mockContacts.slice(0, 3);
  const followUps = mockContacts.filter((c) => c.followUps.length > 0).slice(0, 3);

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
        <View style={{ flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between", paddingTop: 56, paddingBottom: 28 }}>
          <View>
            <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.textMuted, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 4 }}>
              {getGreeting()}
            </Text>
            <Text style={{ fontSize: 26, fontFamily: "Syne_800ExtraBold", color: t.text, lineHeight: 32 }}>
              {user?.name?.split(" ")[0] || "User"}
            </Text>
          </View>
          <Pressable
            onPress={() => router.push("/profile")}
            style={{
              width: 44,
              height: 44,
              borderRadius: 14,
              backgroundColor: t.accentDim,
              borderWidth: 1,
              borderColor: t.accent + "40",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "row",
              marginTop: 8,
            }}
          >
            <Image source={ kalyanImg } style={{ width: 44, height: 44, borderRadius: 14 }} />
          </Pressable>
        </View>

        {/* Stats Row */}
        <View style={{ flexDirection: "row", gap: 10, marginBottom: 24 }}>
          {[
            { label: "People Met", value: "47", color: t.accent },
            { label: "Events", value: "8", color: t.green },
            { label: "Follow-ups", value: "12", color: t.orange },
          ].map((stat) => (
            <View
              key={stat.label}
              style={{
                flex: 1,
                alignItems: "center",
                paddingVertical: 16,
                paddingHorizontal: 8,
                backgroundColor: t.card,
                borderRadius: 16,
                borderWidth: 1,
                borderColor: t.border,
              }}
            >
              <Text style={{ fontSize: 28, fontFamily: "Syne_800ExtraBold", color: stat.color, lineHeight: 34, marginBottom: 4 }}>
                {stat.value}
              </Text>
              <Text style={{ fontSize: 10, fontFamily: "Poppins_400Regular", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1 }}>
                {stat.label}
              </Text>
            </View>
          ))}
        </View>

        {/* Event Mode Button */}
        <Pressable
          onPress={() => router.push("/event-mode")}
          style={{ borderRadius: 20, marginBottom: 28, overflow: "hidden" }}
        >
          <LinearGradient
            colors={[t.accent, t.purple]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              padding: 20,
              borderRadius: 20,
            }}
          >
            <View style={{ flex: 1 }}>
              <Text style={{ fontSize: 16, fontFamily: "Syne_700Bold", color: "#fff", marginBottom: 4 }}>
                Activate Event Mode
              </Text>
              <Text style={{ fontSize: 12, fontFamily: "Poppins_400Regular", color: "rgba(255,255,255,0.7)" }}>
                Start listening for new connections
              </Text>
            </View>
            <View style={{
              width: 44,
              height: 44,
              borderRadius: 14,
              backgroundColor: "rgba(255,255,255,0.18)",
              alignItems: "center",
              justifyContent: "center",
            }}>
              <MicIcon size={20} color="#fff" />
            </View>
          </LinearGradient>
        </Pressable>

        {/* Recent Contacts */}
        <View style={{ marginBottom: 24 }}>
          <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1.2 }}>
              Recent Contacts
            </Text>
            <Pressable onPress={() => router.push("/vault")}>
              <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: t.accent }}>
                View all
              </Text>
            </Pressable>
          </View>

          <View style={{
            backgroundColor: t.card,
            borderRadius: 20,
            borderWidth: 1,
            borderColor: t.border,
            overflow: "hidden",
          }}>
            {recentContacts.map((c, i) => {
              const { score, color } = contactScore(c);
              return (
                <Pressable
                  key={c.id}
                  onPress={() => router.push({ pathname: "/contact", params: { id: String(c.id) } })}
                  style={({ pressed }) => ({
                    flexDirection: "row",
                    alignItems: "center",
                    paddingVertical: 14,
                    paddingHorizontal: 16,
                    backgroundColor: pressed ? t.cardHighlight : "transparent",
                    borderTopWidth: i > 0 ? 1 : 0,
                    borderTopColor: t.border,
                  })}
                >
                  <View style={{
                    width: 42,
                    height: 42,
                    borderRadius: 13,
                    backgroundColor: c.color + "1A",
                    borderWidth: 1.5,
                    borderColor: c.color + "50",
                    alignItems: "center",
                    justifyContent: "center",
                    marginRight: 12,
                  }}>
                    {/* <PersonCircle/> */}
                    {/* <Text style={{ fontSize: 13, fontFamily: "Syne_700Bold", color: c.color }}>
                      {c.avatar}
                    </Text> */}
                    <Image source={c.img ? { uri: c.img } : defaultAvatar} style={{ width: 42, height: 42, borderRadius: 13 }} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={{ fontSize: 14, fontFamily: "Syne_700Bold", color: t.text }}>
                      {c.name}
                    </Text>
                    <Text style={{ fontSize: 12, fontFamily: "Poppins_400Regular", color: t.textMuted, marginTop: 1 }} numberOfLines={1}>
                      {c.role} · {c.company}
                    </Text>
                  </View>
                  <View style={{
                    paddingHorizontal: 10,
                    paddingVertical: 4,
                    borderRadius: 8,
                    backgroundColor: color + "1A",
                  }}>
                    <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: color }}>
                      {score}%
                    </Text>
                  </View>
                </Pressable>
              );
            })}
          </View>
        </View>

        {/* Follow-up Queue */}
        <View>
          <Text style={{ fontSize: 12, fontFamily: "Poppins_600SemiBold", color: t.textMuted, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 14 }}>
            Follow-up Queue
          </Text>
          <View style={{
            backgroundColor: t.card,
            borderRadius: 20,
            borderWidth: 1,
            borderColor: t.border,
            overflow: "hidden",
          }}>
            {followUps.map((c, i) => (
              <View
                key={c.id}
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  paddingVertical: 14,
                  paddingHorizontal: 16,
                  borderTopWidth: i > 0 ? 1 : 0,
                  borderTopColor: t.border,
                  gap: 12,
                }}
              >
                <View style={{
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: t.orange,
                }} />
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.text }}>
                    <Text style={{ fontFamily: "Poppins_600SemiBold", color: t.accent }}>{c.name}</Text>
                    {" — "}{c.followUps[0]}
                  </Text>
                </View>
                <Text style={{ fontSize: 11, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
                  Today
                </Text>
              </View>
            ))}
          </View>
        </View>
      </ScrollView>
    </View>
  );
}
