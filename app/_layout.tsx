import { useTheme } from "@/constants/theme";
import { AuthProvider, useAuth } from "@/services/authContext";
import { Poppins_400Regular, Poppins_600SemiBold } from "@expo-google-fonts/poppins";
import { Syne_700Bold, Syne_800ExtraBold } from "@expo-google-fonts/syne";
import { useFonts } from "expo-font";
import { Stack, useRouter, useSegments } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { StatusBar } from "expo-status-bar";
import React, { useEffect } from "react";
import { ActivityIndicator, View } from "react-native";
import "../global.css";

SplashScreen.preventAutoHideAsync();

function AuthGate({ children }: Readonly<{ children: React.ReactNode }>) {
  const { user, isLoading } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === "(auth)";

    if (!user && !inAuthGroup) {
      router.replace("/(auth)/sign-up");
    } else if (user && inAuthGroup) {
      router.replace("/(tabs)");
    }
  }, [user, isLoading, segments]);

  if (isLoading) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return <>{children}</>;
}

export default function RootLayout() {
  const t = useTheme();

  const [fontsLoaded, fontError] = useFonts({
    Syne_700Bold,
    Syne_800ExtraBold,
    Poppins_400Regular,
    Poppins_600SemiBold,
  });

  useEffect(() => {
    if (fontsLoaded || fontError) {
      SplashScreen.hideAsync();
    }
  }, [fontsLoaded, fontError]);

  if (!fontsLoaded && !fontError) {
    return null;
  }

  return (
    <AuthProvider>
      <StatusBar style="auto" />
      <AuthGate>
        <Stack
          screenOptions={{
            headerShown: false,
            contentStyle: { backgroundColor: t.bg },
            animation: "slide_from_right",
          }}
        >
          <Stack.Screen name="(auth)" />
          <Stack.Screen name="(tabs)" />
          <Stack.Screen
            name="contact"
            options={{
              headerShown: true,
              headerTitle: "",
              headerStyle: { backgroundColor: t.bg },
              headerShadowVisible: false,
              headerTintColor: t.accent,
            }}
          />
          <Stack.Screen
            name="processing"
            options={{
              presentation: "fullScreenModal",
              gestureEnabled: false,
            }}
          />
          <Stack.Screen
            name="talk-bot"
            options={{
              headerShown: true,
              headerTitle: "Talk to Memora",
              headerStyle: { backgroundColor: t.bg },
              headerShadowVisible: false,
              headerTintColor: t.accent,
              headerTitleStyle: {
                fontFamily: "Syne_700Bold",
                color: t.text,
                fontSize: 17,
              },
            }}
          />
        </Stack>
      </AuthGate>
    </AuthProvider>
  );
}
