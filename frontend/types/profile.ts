export type UserProfile = {
  id: string
  fullName: string
  username: string
  email: string
  bio?: string
  role?: string
  company?: string
  interests?: string[]
  skills?: string[]
}