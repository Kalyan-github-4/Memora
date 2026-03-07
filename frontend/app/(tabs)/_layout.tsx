import BottomNav from "@/components/navigation/BottomNav";
import type { BottomTabBarProps } from "@react-navigation/bottom-tabs";
import { Tabs } from "expo-router";

const renderTabBar = (props: BottomTabBarProps) => <BottomNav {...props} />;

export default function TabLayout() {

  return (
    <Tabs
      tabBar={renderTabBar}
      screenOptions={{
        headerShown: false,
      }}
    >
      <Tabs.Screen name="index" />
      <Tabs.Screen name="vault" />
      <Tabs.Screen name="notifications" />
    </Tabs>
  );
}
