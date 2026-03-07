import { useTheme } from "@/constants/theme";
import { Bell } from "lucide-react-native";
import { StatusBar } from "expo-status-bar";
import { Text, View } from "react-native";

export default function NotificationsScreen() {
  const t = useTheme();
  return (
    <View style={{ flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: t.bg, gap: 12 }}>
      <StatusBar style="auto" />
      <View style={{
        width: 56,
        height: 56,
        borderRadius: 18,
        backgroundColor: t.accentDim,
        borderWidth: 1,
        borderColor: t.accent + "40",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: 4,
      }}>
        <Bell size={24} color={t.accent} />
      </View>
      <Text style={{ fontSize: 16, fontFamily: "Syne_700Bold", color: t.text }}>
        Notifications
      </Text>
      <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
        You&apos;re all caught up
      </Text>
    </View>
  );
}
