/**
 * NetworkMemory Backend - Express Server
 *
 * This is the PRIMARY backend for NetworkMemory AI.
 * - Handles all database operations
 * - Orchestrates calls to Python AI service
 * - Serves the React Native frontend
 */

import express from 'express';
import cors from 'cors';
import * as dotenv from 'dotenv';
import { createServer } from 'http';
import { testConnection } from './db/index.js';
import { initializeSocket } from './services/socket.js';

// Load environment variables
dotenv.config();

const app = express();
const httpServer = createServer(app);
const PORT = process.env.PORT || 3000;

// ============================================
// Middleware
// ============================================

// CORS - Allow requests from React Native and Python service
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS === '*' ? '*' : process.env.ALLOWED_ORIGINS?.split(','),
  credentials: true
}));

// JSON body parsing
app.use(express.json());

// Request logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// ============================================
// Basic Routes
// ============================================

/**
 * Root endpoint - API information
 */
app.get('/', (req, res) => {
  res.json({
    service: 'NetworkMemory Backend',
    version: '1.0.0',
    status: 'running',
    timestamp: new Date().toISOString(),
    endpoints: {
      health: '/health',
      audio_processing: '/api/audio/process',
      file_upload: '/api/upload/audio',
      upload_and_process: '/api/upload/process',
      contacts: '/api/contacts',
      contact_detail: '/api/contacts/:id',
      user_profile: '/api/users/me',
      chat: '/api/chat/message',
      elevenlabs_webhook: '/api/elevenlabs/webhook',
      elevenlabs_health: '/api/elevenlabs/health'
    }
  });
});

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    database: 'connected',
    python_service: process.env.PYTHON_AI_URL,
    timestamp: new Date().toISOString()
  });
});

// ============================================
// API Routes
// ============================================

import audioRoutes from './routes/audio.js';
import contactsRoutes from './routes/contacts.js';
import uploadRoutes from './routes/upload.js';
import chatRoutes from './routes/chat.js';
import usersRoutes from './routes/users.js';
import elevenlabsRoutes from './routes/elevenlabs.js';

app.use('/api/audio', audioRoutes);
app.use('/api/contacts', contactsRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/elevenlabs', elevenlabsRoutes);

// ============================================
// Error Handling Middleware
// ============================================

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.url,
    method: req.method
  });
});

// Global error handler
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('[ERROR]', err);

  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

// ============================================
// Server Startup
// ============================================

async function startServer() {
  console.log('\n' + '='.repeat(70));
  console.log('STARTING NETWORKMEMORY BACKEND');
  console.log('='.repeat(70));

  // Test database connection
  console.log('\n[1/3] Testing database connection...');
  const dbConnected = await testConnection();

  if (!dbConnected) {
    console.error('[ERROR] Failed to connect to database. Server will not start.');
    process.exit(1);
  }

  // Initialize Socket.io
  console.log('\n[2/3] Initializing Socket.io...');
  initializeSocket(httpServer);
  console.log('[SOCKET] Socket.io initialized successfully');

  // Start HTTP server (with Socket.io attached)
  console.log('\n[3/3] Starting HTTP server...');

  httpServer.listen(PORT, () => {
    console.log('\n' + '='.repeat(70));
    console.log('SERVER READY');
    console.log('='.repeat(70));
    console.log(`API: http://localhost:${PORT}`);
    console.log(`Health: http://localhost:${PORT}/health`);
    console.log(`Socket.io: ws://localhost:${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`Python AI Service: ${process.env.PYTHON_AI_URL}`);
    console.log('='.repeat(70) + '\n');
  });
}

// Start the server
startServer().catch((error) => {
  console.error('Fatal error starting server:', error);
  process.exit(1);
});

export default app;
