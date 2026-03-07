import { View, Pressable } from "react-native";
import { Image } from "expo-image";
import * as ImagePicker from "expo-image-picker";
import { useState } from "react";
import deafaultImage from "@/assets/images/default-image.jpeg";

export function Avatar({ name }: Readonly<{ name: string }>) {
  const [image, setImage] = useState<string | null>(null);

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      alert("Permission to access media library is required!");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
      allowsEditing: true,
      aspect: [1, 1],
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  } 

  return (
    <Pressable onPress={pickImage}>
      <View
        style={{
          width: 90,
          height: 90,
          borderRadius: 45,
          overflow: "hidden",
          backgroundColor: "#444",
        }}
      >
        <Image
          source={image ? { uri: image } : deafaultImage}
          style={{ width: "100%", height: "100%" }}
          contentFit="cover"
        />
      </View>
    </Pressable>
  );
}