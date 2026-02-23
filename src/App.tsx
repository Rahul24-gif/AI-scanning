/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Camera, Lock, Unlock, AlertTriangle, Crosshair, Volume2, VolumeX, Info, MoveLeft, MoveRight, MoveUp, MoveDown } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Types ---
interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

interface LockedObject {
  class: string;
  bbox: [number, number, number, number];
  lastSeen: number;
}

interface SmoothingState {
  x: number;
  y: number;
  w: number;
  h: number;
  vx: number;
  vy: number;
  vw: number;
  vh: number;
}

// --- Helpers ---
const calculateIoU = (box1: [number, number, number, number], box2: [number, number, number, number]) => {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  const xA = Math.max(x1, x2);
  const yA = Math.max(y1, y2);
  const xB = Math.min(x1 + w1, x2 + w2);
  const yB = Math.min(y1 + h1, y2 + h2);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const box1Area = w1 * h1;
  const box2Area = w2 * h2;

  return interArea / (box1Area + box2Area - interArea);
};

// --- Constants ---
const CONFIDENCE_THRESHOLD = 0.5;
const RECOVERY_THRESHOLD = 0.7;
const SMOOTHING_FACTOR = 0.2; // Lower = smoother but laggier
const ALARM_TIMEOUT = 1000; // ms before alarm triggers if object lost

export default function App() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);
  const requestRef = useRef<number>(null);
  const smoothingRef = useRef<SmoothingState | null>(null);

  // State
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [lockedObject, setLockedObject] = useState<LockedObject | null>(null);
  const [status, setStatus] = useState<'scanning' | 'locked' | 'alarm' | 'recovering'>('scanning');
  const [isMuted, setIsMuted] = useState(false);
  const [detections, setDetections] = useState<DetectedObject[]>([]);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [guidance, setGuidance] = useState<string | null>(null);

  // Refs for tracking state to avoid stale closures in detection loop
  const lockedObjectRef = useRef<LockedObject | null>(null);
  const statusRef = useRef<'scanning' | 'locked' | 'alarm' | 'recovering'>('scanning');

  useEffect(() => {
    lockedObjectRef.current = lockedObject;
  }, [lockedObject]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // --- Initialization ---

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const model = await cocoSsd.load();
        modelRef.current = model;
        setIsModelLoading(false);
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    };
    loadModel();

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      stopAlarm();
    };
  }, []);

  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        if (videoRef.current && canvasRef.current) {
          const { videoWidth, videoHeight } = videoRef.current;
          setDimensions({ width: videoWidth, height: videoHeight });
          canvasRef.current.width = videoWidth;
          canvasRef.current.height = videoHeight;
          setIsCameraActive(true);
          startDetection();
        }
      };
    } catch (error) {
      console.error("Camera access denied:", error);
      alert("Please grant camera access to use this app.");
    }
  };

  // --- Audio Alarm Logic ---

  const initAudio = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
  };

  const startAlarm = () => {
    if (isMuted || !audioContextRef.current) return;
    if (oscillatorRef.current) return;

    const ctx = audioContextRef.current;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.5);
    
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.5);
    oscillatorRef.current = osc;

    osc.onended = () => {
      oscillatorRef.current = null;
      // Re-trigger if still in alarm state
      if (statusRef.current === 'alarm') {
        setTimeout(startAlarm, 100);
      }
    };
  };

  const stopAlarm = () => {
    if (oscillatorRef.current) {
      oscillatorRef.current.stop();
      oscillatorRef.current = null;
    }
  };

  // --- Detection Loop ---

  const startDetection = () => {
    const detect = async () => {
      if (!modelRef.current || !videoRef.current || videoRef.current.paused) {
        requestRef.current = requestAnimationFrame(detect);
        return;
      }

      const predictions = await modelRef.current.detect(videoRef.current);
      setDetections(predictions);
      processTracking(predictions);
      draw(predictions);

      requestRef.current = requestAnimationFrame(detect);
    };
    detect();
  };

  const processTracking = (predictions: DetectedObject[]) => {
    const currentLocked = lockedObjectRef.current;
    if (!currentLocked) {
      setStatus('scanning');
      setGuidance(null);
      return;
    }

    const now = Date.now();
    const dt = Math.min(now - currentLocked.lastSeen, 100) / 1000; // time delta in seconds, max 100ms

    // Predict next position based on velocity if we have smoothing state
    let predictedBox = [...currentLocked.bbox] as [number, number, number, number];
    if (smoothingRef.current) {
       predictedBox = [
         smoothingRef.current.x + smoothingRef.current.vx * dt,
         smoothingRef.current.y + smoothingRef.current.vy * dt,
         smoothingRef.current.w + smoothingRef.current.vw * dt,
         smoothingRef.current.h + smoothingRef.current.vh * dt,
       ];
    }

    // Find the best match for the locked object
    const matches = predictions.filter(p => p.class === currentLocked.class);
    let bestMatch: DetectedObject | null = null;
    let bestScore = -1;

    if (matches.length > 0) {
      matches.forEach(m => {
        const iou = calculateIoU(predictedBox, m.bbox);
        
        // Center distance
        const cx1 = predictedBox[0] + predictedBox[2] / 2;
        const cy1 = predictedBox[1] + predictedBox[3] / 2;
        const cx2 = m.bbox[0] + m.bbox[2] / 2;
        const cy2 = m.bbox[1] + m.bbox[3] / 2;
        const dist = Math.sqrt(Math.pow(cx1 - cx2, 2) + Math.pow(cy1 - cy2, 2));
        
        // Normalize distance based on object size
        const maxDist = Math.max(predictedBox[2], predictedBox[3], 1);
        const distScore = Math.max(0, 1 - dist / maxDist);

        // Combined score: heavily weight IoU and distance, then confidence
        const score = (iou * 0.5) + (distScore * 0.4) + (m.score * 0.1);

        if (score > 0.3 && score > bestScore) {
          bestScore = score;
          bestMatch = m;
        }
      });
    }

    if (bestMatch && (bestMatch as DetectedObject).score > CONFIDENCE_THRESHOLD) {
      const match = bestMatch as DetectedObject;
      
      // Advanced Smoothing (PD controller-like)
      if (!smoothingRef.current) {
        smoothingRef.current = { 
          x: match.bbox[0], y: match.bbox[1], w: match.bbox[2], h: match.bbox[3],
          vx: 0, vy: 0, vw: 0, vh: 0
        };
      } else {
        const prev = smoothingRef.current;
        const alpha = SMOOTHING_FACTOR; // Position smoothing
        const beta = 0.1; // Velocity smoothing
        
        const newX = prev.x + (match.bbox[0] - prev.x) * alpha;
        const newY = prev.y + (match.bbox[1] - prev.y) * alpha;
        const newW = prev.w + (match.bbox[2] - prev.w) * alpha;
        const newH = prev.h + (match.bbox[3] - prev.h) * alpha;

        if (dt > 0) {
           smoothingRef.current = {
             x: newX, y: newY, w: newW, h: newH,
             vx: prev.vx + ((newX - prev.x) / dt - prev.vx) * beta,
             vy: prev.vy + ((newY - prev.y) / dt - prev.vy) * beta,
             vw: prev.vw + ((newW - prev.w) / dt - prev.vw) * beta,
             vh: prev.vh + ((newH - prev.h) / dt - prev.vh) * beta,
           };
        }
      }

      setLockedObject({
        class: match.class,
        bbox: [smoothingRef.current.x, smoothingRef.current.y, smoothingRef.current.w, smoothingRef.current.h],
        lastSeen: now
      });

      // Check boundaries
      const { x, y, w, h } = smoothingRef.current;
      const margin = 0; // Allow tracking right up to the edge
      const isOutOfBounds = x < margin || y < margin || (x + w) > (dimensions.width - margin) || (y + h) > (dimensions.height - margin);

      if (isOutOfBounds) {
        setStatus('alarm');
        updateGuidance(x, y, w, h);
      } else if (statusRef.current === 'alarm' && match.score > RECOVERY_THRESHOLD) {
        setStatus('locked');
        setGuidance(null);
      } else if (statusRef.current !== 'alarm') {
        setStatus('locked');
        setGuidance(null);
      }
    } else {
      // Object lost - predict position
      if (smoothingRef.current && dt > 0 && dt < 0.5) { // Only predict for up to 500ms
         // Apply friction to velocity
         smoothingRef.current.vx *= 0.9;
         smoothingRef.current.vy *= 0.9;
         smoothingRef.current.vw *= 0.9;
         smoothingRef.current.vh *= 0.9;

         smoothingRef.current.x += smoothingRef.current.vx * dt;
         smoothingRef.current.y += smoothingRef.current.vy * dt;
         smoothingRef.current.w += smoothingRef.current.vw * dt;
         smoothingRef.current.h += smoothingRef.current.vh * dt;

         setLockedObject({
           ...currentLocked,
           bbox: [smoothingRef.current.x, smoothingRef.current.y, smoothingRef.current.w, smoothingRef.current.h]
         });

         // Check boundaries even while predicting
         const { x, y, w, h } = smoothingRef.current;
         const margin = 0;
         const isOutOfBounds = x < margin || y < margin || (x + w) > (dimensions.width - margin) || (y + h) > (dimensions.height - margin);
         
         if (isOutOfBounds) {
           setStatus('alarm');
           updateGuidance(x, y, w, h);
         }
      }

      if (now - currentLocked.lastSeen > ALARM_TIMEOUT) {
        setStatus('alarm');
        setGuidance("Object Lost - Searching...");
      }
    }
  };

  const updateGuidance = (x: number, y: number, w: number, h: number) => {
    const centerX = x + w / 2;
    const centerY = y + h / 2;
    const frameCenterX = dimensions.width / 2;
    const frameCenterY = dimensions.height / 2;

    const dx = centerX - frameCenterX;
    const dy = centerY - frameCenterY;

    if (Math.abs(dx) > Math.abs(dy)) {
      setGuidance(dx > 0 ? "Move Left" : "Move Right");
    } else {
      setGuidance(dy > 0 ? "Move Up" : "Move Down");
    }
  };

  const draw = (predictions: DetectedObject[]) => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx || !canvasRef.current) return;

    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Draw all detections faintly
    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      ctx.strokeStyle = '#ffffff44';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, width, height);
      
      ctx.fillStyle = '#ffffff44';
      ctx.font = '12px Inter';
      ctx.fillText(`${prediction.class} (${Math.round(prediction.score * 100)}%)`, x, y > 10 ? y - 5 : 10);
    });

    // Draw locked object with emphasis
    if (lockedObject) {
      const [x, y, width, height] = lockedObject.bbox;
      const color = status === 'alarm' ? '#ef4444' : '#10b981';
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.setLineDash([10, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);

      // Corners
      const cornerSize = 20;
      ctx.lineWidth = 6;
      // Top Left
      ctx.beginPath(); ctx.moveTo(x, y + cornerSize); ctx.lineTo(x, y); ctx.lineTo(x + cornerSize, y); ctx.stroke();
      // Top Right
      ctx.beginPath(); ctx.moveTo(x + width - cornerSize, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + cornerSize); ctx.stroke();
      // Bottom Left
      ctx.beginPath(); ctx.moveTo(x, y + height - cornerSize); ctx.lineTo(x, y + height); ctx.lineTo(x + cornerSize, y + height); ctx.stroke();
      // Bottom Right
      ctx.beginPath(); ctx.moveTo(x + width - cornerSize, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - cornerSize); ctx.stroke();

      // Label
      ctx.fillStyle = color;
      ctx.fillRect(x, y - 30, width, 30);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 16px Inter';
      ctx.fillText(`LOCKED: ${lockedObject.class.toUpperCase()}`, x + 10, y - 10);
      
      // Additional Info Box below object
      const dist = lockedObject.bbox[3] > 0 ? (0.5 * 500 / lockedObject.bbox[3]).toFixed(1) : '?';
      const conf = Math.round((predictions.find(d => d.class === lockedObject.class)?.score || 0) * 100);
      
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(x, y + height + 5, 150, 45);
      ctx.fillStyle = '#10b981';
      ctx.font = '12px Inter';
      ctx.fillText(`Type: ${lockedObject.class}`, x + 10, y + height + 20);
      ctx.fillText(`Dist: ~${dist}m | Conf: ${conf}%`, x + 10, y + height + 40);
    }
  };

  // --- Interactions ---

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (status === 'locked' || status === 'alarm') {
      setLockedObject(null);
      smoothingRef.current = null;
      setStatus('scanning');
      stopAlarm();
      return;
    }

    initAudio();

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const scaleX = dimensions.width / rect.width;
    const scaleY = dimensions.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Find object containing click
    const clickedObject = detections.find(d => {
      const [x, y, w, h] = d.bbox;
      return clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h;
    });

    if (clickedObject) {
      setLockedObject({
        class: clickedObject.class,
        bbox: clickedObject.bbox,
        lastSeen: Date.now()
      });
      smoothingRef.current = { 
        x: clickedObject.bbox[0], y: clickedObject.bbox[1], w: clickedObject.bbox[2], h: clickedObject.bbox[3],
        vx: 0, vy: 0, vw: 0, vh: 0
      };
      setStatus('locked');
    }
  };

  useEffect(() => {
    if (status === 'alarm') {
      startAlarm();
    } else {
      stopAlarm();
    }
  }, [status, isMuted]);

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans selection:bg-emerald-500/30 overflow-hidden flex flex-col">
      {/* Header */}
      <header className="p-4 border-b border-white/10 flex items-center justify-between bg-neutral-900/50 backdrop-blur-md z-20">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Crosshair className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">VisionGuard</h1>
            <p className="text-xs text-neutral-400 font-mono uppercase tracking-widest">Object Tracking System v2.0</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button 
            onClick={() => setIsMuted(!isMuted)}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors text-neutral-400"
          >
            {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
          </button>
          <div className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-tighter flex items-center gap-2 ${
            status === 'scanning' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
            status === 'locked' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
            'bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${
              status === 'scanning' ? 'bg-blue-400' :
              status === 'locked' ? 'bg-emerald-400' :
              'bg-red-400'
            }`} />
            {status}
          </div>
        </div>
      </header>

      {/* Main Viewport */}
      <main className="flex-1 relative flex items-center justify-center bg-black overflow-hidden">
        <video 
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover opacity-60 grayscale-[0.5] ${!isCameraActive ? 'hidden' : ''}`}
        />
        <canvas 
          ref={canvasRef}
          onClick={handleCanvasClick}
          className={`absolute inset-0 w-full h-full object-cover cursor-crosshair z-10 ${!isCameraActive ? 'hidden' : ''}`}
        />

        {!isCameraActive && (
          <div className="flex flex-col items-center gap-6 p-8 text-center max-w-md z-30">
            <div className="w-20 h-20 rounded-full bg-neutral-900 flex items-center justify-center border border-white/10">
              <Camera className="text-neutral-500" size={40} />
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Initialize Vision System</h2>
              <p className="text-neutral-400 text-sm">
                VisionGuard uses real-time AI to track objects. Grant camera access to begin scanning your environment.
              </p>
            </div>
            <button 
              onClick={startCamera}
              disabled={isModelLoading}
              className="w-full py-4 bg-emerald-500 hover:bg-emerald-400 disabled:bg-neutral-800 disabled:text-neutral-600 text-neutral-950 font-bold rounded-2xl transition-all shadow-xl shadow-emerald-500/20 flex items-center justify-center gap-2"
            >
              {isModelLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-neutral-600 border-t-white rounded-full animate-spin" />
                  Loading AI Model...
                </>
              ) : (
                <>
                  <Camera size={20} />
                  Activate Camera
                </>
              )}
            </button>
            <div className="flex items-center gap-2 text-xs text-neutral-500 mt-4">
              <Info size={14} />
              <span>Model: COCO-SSD (TensorFlow.js)</span>
            </div>
          </div>
        )}

        {isCameraActive && (
          <div className="absolute inset-0 pointer-events-none z-20">
            {/* HUD Overlay */}
            {/* Corner Accents */}
            <div className="absolute top-8 left-8 w-12 h-12 border-t-2 border-l-2 border-white/20 rounded-tl-lg" />
            <div className="absolute top-8 right-8 w-12 h-12 border-t-2 border-r-2 border-white/20 rounded-tr-lg" />
            <div className="absolute bottom-8 left-8 w-12 h-12 border-b-2 border-l-2 border-white/20 rounded-bl-lg" />
            <div className="absolute bottom-8 right-8 w-12 h-12 border-b-2 border-r-2 border-white/20 rounded-br-lg" />

            {/* Guidance Indicator */}
            <AnimatePresence>
              {guidance && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.8, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.8, y: 20 }}
                  className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-4"
                >
                  <div className="bg-red-500 text-white p-6 rounded-full shadow-2xl shadow-red-500/40 animate-pulse">
                    {guidance === 'Move Left' && <MoveLeft size={48} />}
                    {guidance === 'Move Right' && <MoveRight size={48} />}
                    {guidance === 'Move Up' && <MoveUp size={48} />}
                    {guidance === 'Move Down' && <MoveDown size={48} />}
                    {guidance === 'Object Lost - Searching...' && <AlertTriangle size={48} />}
                  </div>
                  <div className="bg-black/80 backdrop-blur-md border border-red-500/50 px-6 py-2 rounded-xl">
                    <span className="text-red-400 font-bold uppercase tracking-widest text-sm">{guidance}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Status Bar Bottom */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 w-full max-w-xl px-4">
              <div className="bg-neutral-900/80 backdrop-blur-xl border border-white/10 p-4 rounded-2xl shadow-2xl flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${lockedObject ? 'bg-emerald-500/20 text-emerald-400' : 'bg-white/5 text-neutral-500'}`}>
                    {lockedObject ? <Lock size={24} /> : <Unlock size={24} />}
                  </div>
                  <div>
                    <p className="text-xs text-neutral-500 font-mono uppercase tracking-widest">Target Status</p>
                    <p className="font-bold">
                      {lockedObject ? `Tracking: ${lockedObject.class}` : 'Select target to lock'}
                    </p>
                  </div>
                </div>
                
                {lockedObject && (
                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <p className="text-[10px] text-neutral-500 font-mono uppercase tracking-widest">Proximity</p>
                      <p className="font-bold text-emerald-400">
                        {Math.round((lockedObject.bbox[2] * lockedObject.bbox[3]) / (dimensions.width * dimensions.height) * 100)}%
                      </p>
                    </div>
                    <div className="text-right hidden sm:block">
                      <p className="text-[10px] text-neutral-500 font-mono uppercase tracking-widest">Est. Distance</p>
                      <p className="font-bold text-emerald-400">
                        {/* Rough estimation: assume average object is 0.5m tall, focal length ~500px */}
                        {lockedObject.bbox[3] > 0 ? (0.5 * 500 / lockedObject.bbox[3]).toFixed(1) : '?'}m
                      </p>
                    </div>
                    <div className="text-right hidden sm:block">
                      <p className="text-[10px] text-neutral-500 font-mono uppercase tracking-widest">Confidence</p>
                      <p className="font-bold text-emerald-400">
                        {Math.round((detections.find(d => d.class === lockedObject.class)?.score || 0) * 100)}%
                      </p>
                    </div>
                    <button 
                      onClick={() => {
                        setLockedObject(null);
                        smoothingRef.current = null;
                        setStatus('scanning');
                        stopAlarm();
                      }}
                      className="px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-xs font-bold uppercase transition-colors pointer-events-auto"
                    >
                      Release
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer / Stats */}
      <footer className="p-4 bg-neutral-900/50 border-t border-white/10 flex items-center justify-between text-[10px] text-neutral-500 font-mono uppercase tracking-widest">
        <div className="flex gap-6">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span>Engine: WebGL 2.0</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            <span>FPS: 30 (Stable)</span>
          </div>
        </div>
        <div>
          © 2026 VisionGuard AI Systems
        </div>
      </footer>
    </div>
  );
}
