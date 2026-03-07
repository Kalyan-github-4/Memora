import { Pressable, Text, View } from "react-native";
import { router } from "expo-router";
import { User } from "lucide-react-native";
import { useTheme } from "@/constants/theme";

type Contact = {
  id: number | string;
  name: string;
  role: string;
  company: string;
};

type Props = {
  contact: Contact;
  score: number;
};

export default function Contact({ contact, score }: Props) {
  const t = useTheme();

  return (
    <Pressable
      onPress={() =>
        router.push({
          pathname: "/contact",
          params: { id: String(contact.id) },
        })
      }
      className="bg-memo-card dark:bg-dk-card rounded-3xl py-[14px] px-4 flex-row items-center gap-3.5"
    >
      {/* Avatar */}
      <View
        className="items-center justify-center rounded-full w-11 h-11"
        style={{ backgroundColor: t.bg + "15" }}
      >
        <User size={26} color={t.text} />
      </View>

      {/* Name + Role */}
      <View className="flex-1 min-w-0">
        <Text className="text-sm font-syne-bold text-memo-text dark:text-dk-text">
          {contact.name}
        </Text>

        <Text
          numberOfLines={1}
          className="text-xs font-mono text-memo-textMuted dark:text-dk-textMuted mt-0.5"
        >
          {contact.role} · {contact.company}
        </Text>
      </View>

      {/* Contact Strength */}
      <View className="items-end">
        <Text className="text-[11px] font-mono-bold text-memo-text dark:text-dk-text">
          {score}%
        </Text>

        <Text className="text-[10px] font-mono text-memo-textMuted dark:text-dk-textMuted">
          Contact Strength
        </Text>
      </View>
    </Pressable>
  );
}