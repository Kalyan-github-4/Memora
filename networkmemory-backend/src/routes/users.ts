/**
 * User Profile Routes
 *
 * Handles user profile CRUD operations:
 * - Get user profile
 * - Update/Complete user profile
 * - Create new user (first-time signup)
 */

import { Router, Request, Response } from 'express';
import { db } from '../db/index.js';
import { users } from '../db/schema.js';
import { eq } from 'drizzle-orm';
import { requireAuth, getUserId } from '../middleware/auth.js';

const router = Router();

/**
 * GET /api/users/me
 *
 * Get current user's profile
 *
 * PROTECTED: Requires valid JWT token (Auth0 or Clerk)
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 *
 * Response:
 * {
 *   "user": {
 *     "id": "uuid",
 *     "email": "user@example.com",
 *     "name": "John Doe",
 *     "role": "Software Engineer",
 *     "company": "Tech Corp",
 *     "location": "San Francisco, CA",
 *     "bio": "Passionate about AI and networking",
 *     "industry": "Technology",
 *     "experience": "3-5yrs",
 *     "expertise": ["AI/ML", "Web Development"],
 *     "networkingGoals": ["partnerships", "learning"],
 *     "interests": ["AI", "Blockchain"],
 *     "followUpStyle": "balanced",
 *     "conversationStyle": "professional",
 *     "createdAt": "2024-01-01T00:00:00.000Z"
 *   }
 * }
 */
router.get('/me', requireAuth, async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[USERS] Fetching profile for user: ${userId}`);

    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.id, userId));

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    console.log(`[USERS] Profile found for: ${user.name}`);

    res.json({ user });

  } catch (error: any) {
    console.error('[USERS] Get profile error:', error);
    res.status(500).json({ error: 'Failed to fetch user profile' });
  }
});

/**
 * PUT /api/users/me
 *
 * Update current user's profile
 * This is used for completing the profile during onboarding or updating later
 *
 * PROTECTED: Requires valid JWT token (Auth0 or Clerk)
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 *
 * Request body (all fields optional - send only what you want to update):
 * {
 *   "name": "John Doe",
 *   "role": "Software Engineer",
 *   "company": "Tech Corp",
 *   "location": "San Francisco, CA",
 *   "bio": "Passionate about AI and networking",
 *   "industry": "Technology",
 *   "experience": "3-5yrs",
 *   "expertise": ["AI/ML", "Web Development", "Cloud Computing"],
 *   "networkingGoals": ["partnerships", "learning", "mentoring"],
 *   "interests": ["AI", "Blockchain", "Startups"],
 *   "followUpStyle": "balanced",
 *   "conversationStyle": "professional"
 * }
 *
 * Valid values:
 * - experience: "<1yr" | "1-3yrs" | "3-5yrs" | "5-10yrs" | "10+yrs"
 * - networkingGoals: "hiring" | "job-seeking" | "partnerships" | "fundraising" | "learning" | "mentoring" | "networking" | "clients"
 * - followUpStyle: "quick" | "balanced" | "detailed"
 * - conversationStyle: "casual" | "professional" | "mixed"
 *
 * Response:
 * {
 *   "success": true,
 *   "user": { ... updated user object ... }
 * }
 */
router.put('/me', requireAuth, async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    const updates = req.body;

    console.log(`[USERS] Updating profile for user: ${userId}`);
    console.log('[USERS] Update fields:', Object.keys(updates));

    // Check if user exists
    const [existingUser] = await db
      .select()
      .from(users)
      .where(eq(users.id, userId));

    if (!existingUser) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Validate experience enum if provided
    if (updates.experience && !['<1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs'].includes(updates.experience)) {
      return res.status(400).json({ error: 'Invalid experience value' });
    }

    // Validate followUpStyle enum if provided
    if (updates.followUpStyle && !['quick', 'balanced', 'detailed'].includes(updates.followUpStyle)) {
      return res.status(400).json({ error: 'Invalid followUpStyle value' });
    }

    // Validate conversationStyle enum if provided
    if (updates.conversationStyle && !['casual', 'professional', 'mixed'].includes(updates.conversationStyle)) {
      return res.status(400).json({ error: 'Invalid conversationStyle value' });
    }

    // Update user profile
    const [updatedUser] = await db
      .update(users)
      .set(updates)
      .where(eq(users.id, userId))
      .returning();

    console.log('[USERS] Profile updated successfully');

    res.json({
      success: true,
      user: updatedUser
    });

  } catch (error: any) {
    console.error('[USERS] Update profile error:', error);
    res.status(500).json({ error: 'Failed to update user profile' });
  }
});

/**
 * POST /api/users
 *
 * Create a new user (first-time signup)
 * This should be called after successful authentication with Clerk/Auth0
 *
 * PROTECTED: Requires valid JWT token (Auth0 or Clerk)
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 *
 * Request body:
 * {
 *   "email": "user@example.com",
 *   "name": "John Doe"
 * }
 *
 * Response:
 * {
 *   "success": true,
 *   "user": { ... new user object ... }
 * }
 */
router.post('/', requireAuth, async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    const { email, name } = req.body;

    if (!email || !name) {
      return res.status(400).json({ error: 'email and name are required' });
    }

    console.log(`[USERS] Creating new user: ${email}`);

    // Check if user already exists
    const [existingUser] = await db
      .select()
      .from(users)
      .where(eq(users.id, userId));

    if (existingUser) {
      console.log('[USERS] User already exists, returning existing profile');
      return res.json({
        success: true,
        user: existingUser,
        message: 'User already exists'
      });
    }

    // Create new user
    const [newUser] = await db
      .insert(users)
      .values({
        id: userId,
        email,
        name
      })
      .returning();

    console.log('[USERS] New user created successfully');

    res.status(201).json({
      success: true,
      user: newUser
    });

  } catch (error: any) {
    console.error('[USERS] Create user error:', error);
    res.status(500).json({ error: 'Failed to create user' });
  }
});

/**
 * GET /api/users/:id
 *
 * Get a user by ID (for viewing other users' profiles)
 * Only returns public profile information
 *
 * PROTECTED: Requires valid JWT token (Auth0 or Clerk)
 */
router.get('/:id', requireAuth, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    console.log(`[USERS] Fetching public profile for user: ${id}`);

    const [user] = await db
      .select({
        id: users.id,
        name: users.name,
        role: users.role,
        company: users.company,
        location: users.location,
        bio: users.bio,
        industry: users.industry,
        experience: users.experience,
        expertise: users.expertise,
        interests: users.interests
      })
      .from(users)
      .where(eq(users.id, id));

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json({ user });

  } catch (error: any) {
    console.error('[USERS] Get user error:', error);
    res.status(500).json({ error: 'Failed to fetch user' });
  }
});

export default router;
