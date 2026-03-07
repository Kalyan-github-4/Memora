import { useTheme } from "@/constants/theme";
import type { BottomTabBarProps } from "@react-navigation/bottom-tabs";
import { Bell, Bot, Home, List } from "lucide-react-native";
import { useEffect } from "react";
import { Platform, TouchableOpacity, View } from "react-native";
import Animated, {
    Easing,
    useAnimatedStyle,
    useSharedValue,
    withRepeat,
    withSequence,
    withTiming,
} from "react-native-reanimated";

const TAB_ICONS = {
  index: Home,
  vault: List,
  notifications: Bell,
};

export default function BottomNav({ state, navigation }: Readonly<BottomTabBarProps>) {
  const t = useTheme();

  const scale = useSharedValue(1);
  const opacity = useSharedValue(0.4);

  useEffect(() => {
    scale.value = withRepeat(
      withSequence(
        withTiming(1.12, { duration: 1200, easing: Easing.inOut(Easing.ease) }),
        withTiming(1, { duration: 1200, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );
    opacity.value = withRepeat(
      withSequence(
        withTiming(0.15, { duration: 1200, easing: Easing.inOut(Easing.ease) }),
        withTiming(0.4, { duration: 1200, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );
  }, []);

  const buttonAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const glowAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value * 1.3 }],
    opacity: opacity.value,
  }));

  return (
    <View style={{ position: "absolute", bottom: 24, left: 0, right: 0, alignItems: "center" }}>
      <View style={{ flexDirection: "row", alignItems: "center", width: "90%", maxWidth: 400 }}>
        {/* Main nav bar */}
        <View
          style={[
            {
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-around",
              flex: 1,
              height: 64,
              marginRight: 12,
              borderRadius: 32,
              backgroundColor: t.card,
              borderWidth: 1,
              borderColor: t.border,
            },
            Platform.select({
              ios: {
                shadowColor: "#000",
                shadowOffset: { width: 0, height: 8 },
                shadowOpacity: 0.2,
                shadowRadius: 20,
              },
              android: { elevation: 12 },
            }),
          ]}
        >
          {state.routes.map((route, index) => {
            const Icon = TAB_ICONS[route.name as keyof typeof TAB_ICONS];
            if (!Icon) return null;
            const isFocused = state.index === index;

            const onPress = () => {
              const event = navigation.emit({
                type: "tabPress",
                target: route.key,
                canPreventDefault: true,
              });
              if (!isFocused && !event.defaultPrevented) {
                navigation.navigate(route.name, route.params);
              }
            };

            return (
              <TouchableOpacity
                key={route.key}
                accessibilityRole="button"
                accessibilityState={isFocused ? { selected: true } : undefined}
                onPress={onPress}
                activeOpacity={0.75}
                style={[
                  {
                    alignItems: "center",
                    justifyContent: "center",
                    width: 48,
                    height: 48,
                    borderRadius: 16,
                  },
                  isFocused && {
                    backgroundColor: t.accent + "1A",
                  },
                ]}
              >
                <Icon
                  size={22}
                  color={isFocused ? t.accent : t.textMuted}
                  strokeWidth={isFocused ? 2.2 : 1.8}
                />
              </TouchableOpacity>
            );
          })}
        </View>

        {/* Floating AI Bot Button */}
        <View style={{ position: "relative", alignItems: "center", justifyContent: "center" }}>
          <Animated.View
            style={[
              glowAnimatedStyle,
              {
                position: "absolute",
                width: 56,
                height: 56,
                borderRadius: 28,
                backgroundColor: t.accent,
              },
            ]}
          />
          <Animated.View style={buttonAnimatedStyle}>
            <TouchableOpacity
              style={[
                {
                  width: 56,
                  height: 56,
                  borderRadius: 28,
                  backgroundColor: t.accent,
                  alignItems: "center",
                  justifyContent: "center",
                },
                Platform.select({
                  ios: {
                    shadowColor: t.accent,
                    shadowOffset: { width: 0, height: 6 },
                    shadowOpacity: 0.5,
                    shadowRadius: 14,
                  },
                  android: { elevation: 10 },
                }),
              ]}
              activeOpacity={0.85}
              onPress={() => navigation.navigate("talk-bot")}
            >
              <Bot size={24} color="#fff" />
            </TouchableOpacity>
          </Animated.View>
        </View>
      </View>
    </View>
  );
}
