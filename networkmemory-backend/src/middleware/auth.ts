/**
 * Auth0 JWT Authentication Middleware
 *
 * Validates JWT tokens from mobile app users
 * Extracts user information and custom claims (user_type, privacy_level)
 *
 * Usage:
 *   import { requireAuth, optionalAuth } from './middleware/auth';
 *
 *   // Require authentication
 *   router.get('/protected', requireAuth, handler);
 *
 *   // Optional authentication (includes user info if available)
 *   router.get('/optional', optionalAuth, handler);
 */

import { Request, Response, NextFunction } from 'express';
import { auth, requiredScopes, UnauthorizedError } from 'express-oauth2-jwt-bearer';

// Extend Express Request type to include auth property
declare global {
  namespace Express {
    interface Request {
      auth?: {
        payload: {
          sub: string;           // User ID
          user_type?: string;    // student | team | individual
          privacy_level?: string;  // campus-only | organization | private
          email_domain?: string;
          [key: string]: any;
        };
      };
    }
  }
}

/**
 * Check if Auth0 is configured
 */
function isAuth0Configured(): boolean {
  return !!(
    process.env.AUTH0_ISSUER_BASE_URL &&
    process.env.AUTH0_API_AUDIENCE
  );
}

/**
 * JWT validation middleware (from Auth0)
 *
 * Validates:
 * - Token signature
 * - Token expiration
 * - Issuer
 * - Audience
 */
let validateJWT: any;

try {
  if (isAuth0Configured()) {
    validateJWT = auth({
      issuerBaseURL: process.env.AUTH0_ISSUER_BASE_URL,
      audience: process.env.AUTH0_API_AUDIENCE,
      tokenSigningAlg: 'RS256'
    });
    console.log('[AUTH] Auth0 JWT validation configured');
  } else {
    console.warn('[AUTH] Auth0 not configured - using development mode');
    validateJWT = null;
  }
} catch (error: any) {
  console.error('[AUTH] Failed to configure Auth0:', error.message);
  validateJWT = null;
}

/**
 * Required authentication middleware
 *
 * Blocks requests without valid JWT token
 * In development (if Auth0 not configured), uses mock auth
 */
export const requireAuth = (req: Request, res: Response, next: NextFunction) => {
  // If Auth0 is not configured (development mode)
  if (!validateJWT) {
    if (process.env.NODE_ENV === 'development') {
      // Create mock auth payload for development
      req.auth = {
        payload: {
          sub: req.headers['x-user-id'] as string || req.body.userId || 'dev-user-123',
          user_type: 'individual',
          privacy_level: 'private',
          email_domain: 'example.com'
        }
      };
      console.warn('[AUTH] Development mode - using mock authentication');
      return next();
    } else {
      return res.status(500).json({
        error: 'Authentication not configured',
        details: 'Server misconfiguration - Auth0 credentials missing'
      });
    }
  }

  // Use real Auth0 validation
  validateJWT(req, res, (err: any) => {
    if (err) {
      if (err instanceof UnauthorizedError) {
        return res.status(401).json({
          error: 'Authentication required',
          details: err.message
        });
      }
      return res.status(500).json({
        error: 'Authentication error',
        details: err.message
      });
    }
    next();
  });
};

/**
 * Optional authentication middleware
 *
 * Allows requests through but includes user info if token is present
 */
export const optionalAuth = (req: Request, res: Response, next: NextFunction) => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    // No token provided - allow request but without user context
    return next();
  }

  // Token provided - validate it
  validateJWT(req, res, (err: any) => {
    if (err) {
      // Invalid token - return error
      return res.status(401).json({
        error: 'Invalid authentication token',
        details: err.message
      });
    }
    next();
  });
};

/**
 * Helper: Get user ID from authenticated request
 */
export function getUserId(req: Request): string | null {
  return req.auth?.payload.sub || null;
}

/**
 * Helper: Get user type from authenticated request
 */
export function getUserType(req: Request): 'student' | 'team' | 'individual' | null {
  const userType = req.auth?.payload.user_type;
  if (userType === 'student' || userType === 'team' || userType === 'individual') {
    return userType;
  }
  return null;
}

/**
 * Helper: Get privacy level from authenticated request
 */
export function getPrivacyLevel(req: Request): 'campus-only' | 'organization' | 'private' | null {
  const privacyLevel = req.auth?.payload.privacy_level;
  if (privacyLevel === 'campus-only' || privacyLevel === 'organization' || privacyLevel === 'private') {
    return privacyLevel;
  }
  return null;
}

/**
 * Helper: Check if user has permission to access resource
 *
 * @param req - Express request
 * @param resourceUserId - User ID of resource owner
 * @returns true if user can access, false otherwise
 */
export function canAccessResource(req: Request, resourceUserId: string): boolean {
  const currentUserId = getUserId(req);

  if (!currentUserId) {
    return false;  // Not authenticated
  }

  if (currentUserId === resourceUserId) {
    return true;  // User owns the resource
  }

  // Additional logic for shared resources can go here
  // For now, users can only access their own resources
  return false;
}

/**
 * Middleware: Require specific scopes
 *
 * Usage:
 *   router.post('/process', requireAuth, requireScopes(['process:audio']), handler);
 */
export const requireScope = (scopes: string[]) => {
  return requiredScopes(scopes);
};

