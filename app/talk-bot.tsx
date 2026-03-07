import { useTheme } from "@/constants/theme";
import { StatusBar } from "expo-status-bar";
import { Bot } from "lucide-react-native";
import { Text, View } from "react-native";

export default function TalkBotScreen() {
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
        <Bot size={26} color={t.accent} />
      </View>
      <Text style={{ fontSize: 16, fontFamily: "Syne_700Bold", color: t.text }}>
        Talk to Memora
      </Text>
      <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
        Chat feature coming soon
      </Text>
    </View>
  );
}
