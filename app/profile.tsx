import { useTheme } from "@/constants/theme";
import { useProfile } from "@/hooks/useProfile";
import type {
  ConversationStyle,
  ExperienceLevel,
  FollowUpStyle,
  NetworkingGoal,
} from "@/types";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { StatusBar } from "expo-status-bar";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { ProfileCompletion } from "@/components/profile/ProfileCompletion";
import { Avatar } from "@/components/profile/Avatar";
import { FieldRow } from "@/components/profile/FieldRow";
import { OptionRow } from "@/components/profile/OptionRow";
import { TagRow } from "@/components/profile/TagRow";
import { MultiSelectRow } from "@/components/profile/MultiSelectRow";
import { Card } from "@/components/profile/Card";
import { Divider } from "@/components/profile/Divider";
import { SectionLabel } from "@/components/profile/SectionLabel";

// ─── Constants ────────────────────────────────────────────────────────────────

const EXPERIENCE_OPTIONS: { value: ExperienceLevel; label: string }[] = [
  { value: "<1yr", label: "< 1 yr" },
  { value: "1-3yrs", label: "1–3 yrs" },
  { value: "3-5yrs", label: "3–5 yrs" },
  { value: "5-10yrs", label: "5–10 yrs" },
  { value: "10+yrs", label: "10+ yrs" },
];

const GOAL_OPTIONS: { value: NetworkingGoal; label: string }[] = [
  { value: "hiring", label: "Hiring" },
  { value: "job-seeking", label: "Job seeking" },
  { value: "partnerships", label: "Partnerships" },
  { value: "fundraising", label: "Fundraising" },
  { value: "learning", label: "Learning" },
  { value: "mentoring", label: "Mentoring" },
  { value: "clients", label: "Finding clients" },
  { value: "networking", label: "General networking" },
];

const FOLLOWUP_OPTIONS: { value: FollowUpStyle; label: string }[] = [
  { value: "quick", label: "Quick check-in" },
  { value: "balanced", label: "Balanced update" },
  { value: "detailed", label: "Detailed recap" },
];

const CONVERSATION_OPTIONS: { value: ConversationStyle; label: string }[] = [
  { value: "casual", label: "Casual" },
  { value: "professional", label: "Professional" },
  { value: "mixed", label: "Depends on context" },
];

// ─── Main Screen ──────────────────────────────────────────────────────────────

export default function ProfileScreen() {
  const t = useTheme();
  const { profile, update, save, isDirty, isSaving, isLoading } = useProfile();

  // ─── Profile completion calculation ─────────────────

  const fields = [
    profile.name,
    profile.role,
    profile.company,
    profile.bio,
    profile.location,
    profile.industry,
    profile.experience,
    profile.expertise.length > 0,
    profile.networkingGoals.length > 0,
    profile.interests.length > 0,
  ];

  const completedFields = fields.filter(Boolean).length;
  const completionPercent = Math.round(
    (completedFields / fields.length) * 100
  );

  if (isLoading) {
    return (
      <View style={{ flex: 1, backgroundColor: t.bg, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator color={t.accent} />
      </View>
    );
  }

  const handleSave = () => save(profile);

  const addTag = (field: "expertise" | "interests", tag: string) =>
    update({ [field]: [...profile[field], tag] });

  const removeTag = (field: "expertise" | "interests", tag: string) =>
    update({ [field]: profile[field].filter((t) => t !== tag) });

  return (
    <View style={{ flex: 1, backgroundColor: t.bg }}>
      <StatusBar style="auto" />

      {/* Top glow */}
      <LinearGradient
        colors={[t.accent + "22", t.purple + "10", "transparent"]}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 280,
          pointerEvents: "none",
        }}
      />

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        keyboardVerticalOffset={80}
      >
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={{ paddingBottom: isDirty ? 100 : 40, paddingHorizontal: 20 }}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* ── Nav bar ────────────────────────────── */}
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              paddingTop: 56,
              paddingBottom: 28,
            }}
          >
            <Pressable
              onPress={() => router.back()}
              style={{
                width: 40,
                height: 40,
                borderRadius: 12,
                backgroundColor: t.surface,
                borderWidth: 1,
                borderColor: t.border,
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Text style={{ fontSize: 18, color: t.text, lineHeight: 22 }}>←</Text>
            </Pressable>
            <Text
              style={{
                fontSize: 16,
                fontFamily: "Syne_700Bold",
                color: t.text,
              }}
            >
              Your Profile
            </Text>
            <View style={{ width: 40 }} />
          </View>

          {/* ── Hero card ──────────────────────────── */}
          <View
            style={{
              backgroundColor: t.card,
              borderRadius: 24,
              borderWidth: 1,
              borderColor: t.border,
              padding: 20,
              flexDirection: "row",
              alignItems: "flex-start",
              gap: 16,
              marginBottom: 28,
            }}
          >

            <Avatar name={profile.name || "?"} />
            <View style={{ flex: 1, gap: 2 }}>
              <TextInput
                value={profile.name}
                onChangeText={(v) => update({ name: v })}
                placeholder="Your name"
                placeholderTextColor={t.textSub}
                style={{
                  fontSize: 22,
                  fontFamily: "Syne_800ExtraBold",
                  color: t.text,
                  letterSpacing: -0.3,
                  padding: 0,
                }}
              />
              <TextInput
                value={profile.role}
                onChangeText={(v) => update({ role: v })}
                placeholder="Role / Title"
                placeholderTextColor={t.textSub}
                style={{
                  fontSize: 14,
                  fontFamily: "Poppins_600SemiBold",
                  color: t.accent,
                  padding: 0,
                  marginTop: 2,
                }}
              />
              <TextInput
                value={profile.company}
                onChangeText={(v) => update({ company: v })}
                placeholder="Company"
                placeholderTextColor={t.textSub}
                style={{
                  fontSize: 13,
                  fontFamily: "Poppins_400Regular",
                  color: t.textMuted,
                  padding: 0,
                  marginTop: 1,
                }}
              />
            </View>
          </View>
          <ProfileCompletion percent={completionPercent} />

          {/* ── About ─────────────────────────────── */}
          <SectionLabel title="About" />
          <Card>
            <FieldRow
              label="Bio"
              value={profile.bio}
              onChange={(v) => update({ bio: v })}
              placeholder="A short intro — how you'd describe yourself at an event"
              multiline
            />
            <Divider />
            <FieldRow
              label="Location"
              value={profile.location}
              onChange={(v) => update({ location: v })}
              placeholder="City, Country"
            />
          </Card>

          {/* ── Professional ──────────────────────── */}
          <SectionLabel title="Professional" />
          <Card>
            <FieldRow
              label="Industry"
              value={profile.industry}
              onChange={(v) => update({ industry: v })}
              placeholder="e.g. FinTech, Healthcare, SaaS"
            />
            <Divider />
            <OptionRow<ExperienceLevel>
              label="Experience"
              options={EXPERIENCE_OPTIONS}
              value={profile.experience}
              onChange={(v) => update({ experience: v })}
            />
            <Divider />
            <TagRow
              label="Expertise"
              tags={profile.expertise}
              onAdd={(tag) => addTag("expertise", tag)}
              onRemove={(tag) => removeTag("expertise", tag)}
              placeholder="e.g. Product, ML, Growth"
              color={t.accent}
            />
          </Card>

          {/* ── Networking Goals ──────────────────── */}
          <SectionLabel title="Networking Goals" />
          <Card>
            <MultiSelectRow<NetworkingGoal>
              label="What are you looking for?"
              options={GOAL_OPTIONS}
              values={profile.networkingGoals}
              onChange={(v) => update({ networkingGoals: v })}
              color={t.purple}
            />
          </Card>

          {/* ── Interests ────────────────────────── */}
          <SectionLabel title="Interests" />
          <Card>
            <TagRow
              label="Topics you care about"
              tags={profile.interests}
              onAdd={(tag) => addTag("interests", tag)}
              onRemove={(tag) => removeTag("interests", tag)}
              placeholder="e.g. AI, Climate, Design"
              color={t.green}
            />
          </Card>

          {/* ── Preferences ──────────────────────── */}
          <SectionLabel title="Preferences" />
          <Card>
            <OptionRow<FollowUpStyle>
              label="Follow-up style"
              options={FOLLOWUP_OPTIONS}
              value={profile.followUpStyle}
              onChange={(v) => update({ followUpStyle: v })}
            />
            <Divider />
            <OptionRow<ConversationStyle>
              label="Conversation style"
              options={CONVERSATION_OPTIONS}
              value={profile.conversationStyle}
              onChange={(v) => update({ conversationStyle: v })}
            />
          </Card>
        </ScrollView>

        {/* ── Sticky save bar ────────────────────── */}
        {isDirty && (
          <View
            style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              paddingHorizontal: 20,
              paddingVertical: 16,
              paddingBottom: Platform.OS === "ios" ? 32 : 16,
              backgroundColor: t.bg,
              borderTopWidth: 1,
              borderTopColor: t.border,
            }}
          >
            <Pressable
              onPress={handleSave}
              disabled={isSaving}
              style={({ pressed }) => ({
                backgroundColor: t.accent,
                borderRadius: 16,
                paddingVertical: 15,
                alignItems: "center",
                opacity: pressed || isSaving ? 0.75 : 1,
              })}
            >
              {isSaving ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text
                  style={{
                    fontSize: 15,
                    fontFamily: "Syne_700Bold",
                    color: "#fff",
                    letterSpacing: 0.2,
                  }}
                >
                  Save Changes
                </Text>
              )}
            </Pressable>
          </View>
        )}
      </KeyboardAvoidingView>
    </View>
  );
}
