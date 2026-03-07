/**
 * Auth0 Machine-to-Machine (M2M) Authentication Service
 *
 * THE SHOWCASE FEATURE for Auth0 Hackathon Track
 *
 * This service secures communication between Node.js backend and Python AI service
 * using Auth0's M2M authentication (Client Credentials Flow).
 *
 * Key Innovation:
 * - AI service only accepts requests with valid Auth0 tokens
 * - Automatic token management (fetch, cache, refresh)
 * - Enterprise-grade security for AI operations
 *
 * Flow:
 * 1. Node.js gets M2M token from Auth0
 * 2. Node.js includes token when calling Python AI service
 * 3. Python AI validates token before processing
 */

import axios from 'axios';

interface M2MToken {
  access_token: string;
  expires_in: number;
  token_type: string;
  scope: string;
}

interface CachedToken {
  token: string;
  expiresAt: number;  // Unix timestamp
}

class Auth0M2MService {
  private cachedToken: CachedToken | null = null;

  private readonly domain: string;
  private readonly clientId: string;
  private readonly clientSecret: string;
  private readonly audience: string;

  constructor() {
    // Load from environment variables
    this.domain = process.env.AUTH0_DOMAIN || '';
    this.clientId = process.env.AUTH0_M2M_CLIENT_ID || '';
    this.clientSecret = process.env.AUTH0_M2M_CLIENT_SECRET || '';
    this.audience = process.env.AUTH0_M2M_AUDIENCE || '';

    // Validate configuration
    if (!this.domain || !this.clientId || !this.clientSecret || !this.audience) {
      console.warn('[AUTH0-M2M] Auth0 M2M not configured - AI service calls will be unauthenticated');
    }
  }

  /**
   * Check if Auth0 M2M is properly configured
   */
  public isConfigured(): boolean {
    return !!(this.domain && this.clientId && this.clientSecret && this.audience);
  }

  /**
   * Get a valid M2M access token
   *
   * Automatically handles:
   * - Fetching new token from Auth0
   * - Caching token until expiration
   * - Refreshing expired tokens
   *
   * @returns Access token to use for Python AI service calls
   */
  public async getAccessToken(): Promise<string> {
    // Check if we have a valid cached token
    if (this.cachedToken && this.cachedToken.expiresAt > Date.now()) {
      console.log('[AUTH0-M2M] Using cached token');
      return this.cachedToken.token;
    }

    // Fetch new token from Auth0
    console.log('[AUTH0-M2M] Fetching new M2M token from Auth0...');

    try {
      const response = await axios.post<M2MToken>(
        `https://${this.domain}/oauth/token`,
        {
          grant_type: 'client_credentials',
          client_id: this.clientId,
          client_secret: this.clientSecret,
          audience: this.audience
        },
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );

      const { access_token, expires_in } = response.data;

      // Cache token (expire 60 seconds early to be safe)
      const expiresAt = Date.now() + (expires_in - 60) * 1000;
      this.cachedToken = {
        token: access_token,
        expiresAt: expiresAt
      };

      console.log('[AUTH0-M2M] Successfully obtained M2M token');
      console.log(`[AUTH0-M2M] Token expires in ${expires_in} seconds`);

      return access_token;

    } catch (error: any) {
      console.error('[AUTH0-M2M] Failed to get M2M token:', error.message);

      if (error.response) {
        console.error('[AUTH0-M2M] Auth0 error:', error.response.data);
      }

      throw new Error('Failed to obtain M2M authentication token');
    }
  }

  /**
   * Clear cached token (useful for testing or error recovery)
   */
  public clearCache(): void {
    console.log('[AUTH0-M2M] Clearing token cache');
    this.cachedToken = null;
  }

  /**
   * Get token info (for debugging)
   */
  public getTokenInfo(): { hasToken: boolean; expiresIn?: number } {
    if (!this.cachedToken) {
      return { hasToken: false };
    }

    const expiresIn = Math.max(0, this.cachedToken.expiresAt - Date.now()) / 1000;
    return {
      hasToken: true,
      expiresIn: Math.round(expiresIn)
    };
  }
}

// Export singleton instance
export const auth0M2M = new Auth0M2MService();

/**
 * Helper function: Make authenticated request to Python AI service
 *
 * This wraps axios calls with automatic Auth0 M2M token injection
 *
 * @param url - Python AI service endpoint
 * @param data - Request payload
 * @param timeout - Request timeout in milliseconds
 * @returns Response from Python AI service
 */
export async function callPythonAIService<T = any>(
  url: string,
  data: any,
  timeout: number = 120000
): Promise<T> {
  const pythonUrl = process.env.PYTHON_AI_URL || 'http://localhost:8000';
  const fullUrl = `${pythonUrl}${url}`;

  // Get M2M token if configured
  let headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  if (auth0M2M.isConfigured()) {
    try {
      const token = await auth0M2M.getAccessToken();
      headers['Authorization'] = `Bearer ${token}`;
      console.log('[AUTH0-M2M] Added authentication to Python AI request');
    } catch (error) {
      console.error('[AUTH0-M2M] Failed to get token, proceeding without auth');
    }
  } else {
    console.warn('[AUTH0-M2M] Auth0 M2M not configured - calling Python without authentication');
  }

  // Make authenticated request
  const response = await axios.post(fullUrl, data, {
    headers,
    timeout
  });

  return response.data;
}
