import { useTheme } from "@/constants/theme";
import { Pressable, Text, View } from "react-native";

type Props = {
  title?: string;
  onViewAll?: () => void;
  children: React.ReactNode;
};

export default function ContactsInfoContainer({
  title = "Statistics",
  onViewAll,
  children,
}: Props) {
  const t = useTheme();

  return (
    <View
      style={{
        backgroundColor: "#F6F8FB",
        borderTopLeftRadius: 32,
        borderTopRightRadius: 32,
        paddingHorizontal: 20,
        paddingTop: 20,
        paddingBottom: 30,

        shadowColor: "#000",
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.12,
        shadowRadius: 20,
        elevation: 20,
      }}
    >
      {/* Header */}
      <View
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 18,
        }}
      >
        <Text
          style={{
            fontSize: 18,
            fontFamily: "Syne_700Bold",
            color: t.text,
          }}
        >
          {title}
        </Text>

        {onViewAll && (
          <Pressable onPress={onViewAll}>
            <Text
              style={{
                fontSize: 13,
                fontFamily: "Poppins_400Regular",
                color: t.accent,
              }}
            >
              View All
            </Text>
          </Pressable>
        )}
      </View>

      {children}
    </View>
  );
}