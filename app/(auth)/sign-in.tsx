import { useAuth } from "@/services/authContext";
import { useTheme } from "@/constants/theme";
import { LinearGradient } from "expo-linear-gradient";
import { Link, useRouter } from "expo-router";
import React, { useState } from "react";
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

export default function SignInScreen() {
  const t = useTheme();
  const { signIn } = useAuth();
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSignIn = async () => {
    setError("");
    setLoading(true);
    const result = await signIn(email, password);
    setLoading(false);
    if (!result.success) {
      setError(result.error || "Sign in failed");
      return;
    }
    router.replace("/(tabs)");
  };

  const canSubmit = email.trim().length > 0 && password.trim().length > 0 && !loading;

  return (
    <View style={{ flex: 1, backgroundColor: t.bg }}>
      <LinearGradient
        colors={[t.accent + "28", t.purple + "14", "transparent"]}
        style={{ position: "absolute", top: 0, left: 0, right: 0, height: 350, pointerEvents: "none" }}
      />
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1, justifyContent: "center", paddingHorizontal: 28, paddingBottom: 40 }}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Logo / Brand */}
          <View style={{ alignItems: "center", marginBottom: 48 }}>
            <View
              style={{
                width: 72,
                height: 72,
                borderRadius: 22,
                backgroundColor: t.accent + "18",
                borderWidth: 1,
                borderColor: t.accent + "40",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: 20,
              }}
            >
              <Text style={{ fontSize: 32, fontFamily: "Syne_800ExtraBold", color: t.accent }}>M</Text>
            </View>
            <Text style={{ fontSize: 28, fontFamily: "Syne_800ExtraBold", color: t.text, marginBottom: 6 }}>
              Welcome back
            </Text>
            <Text style={{ fontSize: 14, fontFamily: "Poppins_400Regular", color: t.textMuted, textAlign: "center" }}>
              Sign in to continue to Memora
            </Text>
          </View>

          {/* Form */}
          <View style={{ gap: 16 }}>
            <View>
              <Text style={{ fontSize: 13, fontFamily: "Poppins_600SemiBold", color: t.textMuted, marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
                Email
              </Text>
              <TextInput
                style={{
                  backgroundColor: t.card,
                  borderWidth: 1,
                  borderColor: t.border,
                  borderRadius: 14,
                  paddingHorizontal: 16,
                  paddingVertical: 14,
                  fontSize: 16,
                  fontFamily: "Poppins_400Regular",
                  color: t.text,
                }}
                value={email}
                onChangeText={setEmail}
                placeholder="you@email.com"
                placeholderTextColor={t.textSub}
                autoCapitalize="none"
                keyboardType="email-address"
                autoComplete="email"
              />
            </View>

            <View>
              <Text style={{ fontSize: 13, fontFamily: "Poppins_600SemiBold", color: t.textMuted, marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
                Password
              </Text>
              <TextInput
                style={{
                  backgroundColor: t.card,
                  borderWidth: 1,
                  borderColor: t.border,
                  borderRadius: 14,
                  paddingHorizontal: 16,
                  paddingVertical: 14,
                  fontSize: 16,
                  fontFamily: "Poppins_400Regular",
                  color: t.text,
                }}
                value={password}
                onChangeText={setPassword}
                placeholder="Enter your password"
                placeholderTextColor={t.textSub}
                secureTextEntry
              />
            </View>

            {error ? (
              <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: "#EF4444", textAlign: "center" }}>
                {error}
              </Text>
            ) : null}

            <Pressable
              onPress={handleSignIn}
              disabled={!canSubmit}
              style={{ borderRadius: 16, overflow: "hidden", marginTop: 8, opacity: canSubmit ? 1 : 0.5 }}
            >
              <LinearGradient
                colors={[t.accent, t.purple]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  paddingVertical: 16,
                  alignItems: "center",
                  borderRadius: 16,
                }}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={{ fontSize: 16, fontFamily: "Syne_700Bold", color: "#fff" }}>
                    Sign In
                  </Text>
                )}
              </LinearGradient>
            </Pressable>
          </View>

          {/* Link to sign-up */}
          <View style={{ flexDirection: "row", justifyContent: "center", alignItems: "center", marginTop: 28, gap: 4 }}>
            <Text style={{ fontSize: 14, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
              Don&apos;t have an account?
            </Text>
            <Link href="/(auth)/sign-up" asChild>
              <Pressable>
                <Text style={{ fontSize: 14, fontFamily: "Poppins_600SemiBold", color: t.accent }}>
                  Sign up
                </Text>
              </Pressable>
            </Link>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}