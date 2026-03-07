import { CheckIcon } from "@/components/ui/icons";
import { mockContacts } from "@/constants/data";
import { useTheme } from "@/constants/theme";
import { router } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useEffect, useRef, useState } from "react";
import { Text, View } from "react-native";

const steps = [
  { label: "Transcribing audio", sub: "Whisper API processing..." },
  { label: "Separating speakers", sub: "Diarization running..." },
  { label: "Extracting contact info", sub: "Claude AI analyzing..." },
  { label: "Building contact card", sub: "Structuring data..." },
  { label: "Saved to your vault", sub: "Done!" },
];

export default function ProcessingScreen() {
  const t = useTheme();
  const [step, setStep] = useState(0);
  const doneRef = useRef(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((s) => {
        if (s >= steps.length - 1) {
          clearInterval(interval);
          if (!doneRef.current) {
            doneRef.current = true;
            setTimeout(() => {
              router.replace({
                pathname: "/contact",
                params: { id: String(mockContacts[0].id) },
              });
            }, 1000);
          }
          return s;
        }
        return s + 1;
      });
    }, 900);
    return () => clearInterval(interval);
  }, []);

  return (
    <View style={{ flex: 1, backgroundColor: t.bg, paddingHorizontal: 28, justifyContent: "center" }}>
      <StatusBar style="auto" />

      <View style={{ alignItems: "center", marginBottom: 48 }}>
        <Text style={{ fontSize: 24, fontFamily: "Syne_800ExtraBold", color: t.text, marginBottom: 8 }}>
          Processing
        </Text>
        <Text style={{ fontSize: 13, fontFamily: "Poppins_400Regular", color: t.textMuted }}>
          Building your contact card...
        </Text>
      </View>

      <View style={{ gap: 10 }}>
        {steps.map((s, i) => {
          const isDone = i < step;
          const isCurrent = i === step;
          const isFuture = i > step + 1;

          return (
            <View
              key={i}
              style={{
                flexDirection: "row",
                alignItems: "center",
                gap: 14,
                paddingVertical: 14,
                paddingHorizontal: 18,
                borderRadius: 16,
                backgroundColor: i <= step ? t.card : "transparent",
                borderWidth: 1,
                borderColor: i <= step ? t.border : "transparent",
                opacity: isFuture ? 0.25 : 1,
              }}
            >
              <View style={{
                width: 28,
                height: 28,
                borderRadius: 28,
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: isDone ? t.green : isCurrent ? t.accent : t.border,
              }}>
                {isDone ? (
                  <CheckIcon size={12} color={t.bg} />
                ) : isCurrent ? (
                  <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: t.bg }} />
                ) : (
                  <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: t.textMuted }} />
                )}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={{
                  fontSize: 14,
                  fontFamily: "Poppins_600SemiBold",
                  color: i <= step ? t.text : t.textMuted,
                }}>
                  {s.label}
                </Text>
                {isCurrent && (
                  <Text style={{ fontSize: 11, fontFamily: "Poppins_400Regular", color: t.textMuted, marginTop: 2 }}>
                    {s.sub}
                  </Text>
                )}
              </View>
            </View>
          );
        })}
      </View>
    </View>
  );
}
