import React, { useEffect, useRef, useCallback, memo } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape, { type Core, type EventObject } from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import dagre from 'cytoscape-dagre';
import fcose from 'cytoscape-fcose';
import cola from 'cytoscape-cola';
import { GRAPH_COLORS } from '../../utils/constants';
import type { CytoscapeNode, CytoscapeEdge } from '../../types';

// Register layout algorithms
cytoscape.use(coseBilkent);
cytoscape.use(dagre);
cytoscape.use(fcose);
cytoscape.use(cola);

interface GraphCanvasProps {
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
  onNodeSelect?: (node: CytoscapeNode | null) => void;
  onNodeHover?: (node: CytoscapeNode | null) => void;
  onReady?: (cy: Core) => void;
  layout?: string;
  className?: string;
}

const GraphCanvas: React.FC<GraphCanvasProps> = memo(({
  nodes,
  edges,
  onNodeSelect,
  onNodeHover,
  onReady,
  layout = 'cose-bilkent',
  className = '',
}) => {
  const cyRef = useRef<Core | null>(null);
  const isInitialized = useRef(false);
  const previousLayout = useRef(layout);

  // ✅ PHASE 3: Enhanced stylesheet with DYNAMIC sizing based on weight
  const stylesheet = [
    // Default node styles - DYNAMIC sizing by weight
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'text-valign': 'bottom' as any,
        'text-halign': 'center' as any,
        'text-margin-y': 4,
        'background-color': (ele: any) => {
          const type = ele.data('type');
          return GRAPH_COLORS[type as keyof typeof GRAPH_COLORS] || GRAPH_COLORS.entity;
        },
        // ✅ PHASE 3: DYNAMIC node size based on weight (20px - 50px range)
        width: (ele: any) => {
          const weight = ele.data('weight') || 0.5;
          const minSize = 20;
          const maxSize = 50;
          // Normalize weight to 0-1 range (assuming max weight ~1.0)
          const normalized = Math.min(weight, 1.0);
          return minSize + (normalized * (maxSize - minSize));
        },
        height: (ele: any) => {
          const weight = ele.data('weight') || 0.5;
          const minSize = 20;
          const maxSize = 50;
          const normalized = Math.min(weight, 1.0);
          return minSize + (normalized * (maxSize - minSize));
        },
        'text-wrap': 'wrap' as any,
        'text-max-width': '80px',
        // ✅ PHASE 3: Dynamic font size based on node size
        'font-size': (ele: any) => {
          const weight = ele.data('weight') || 0.5;
          const minFont = 8;
          const maxFont = 12;
          const normalized = Math.min(weight, 1.0);
          return `${minFont + (normalized * (maxFont - minFont))}px`;
        },
        'font-weight': '600',
        'color': '#111',
        'text-outline-color': '#fff',
        'text-outline-width': 1.5,
        'text-background-color': 'rgba(255, 255, 255, 0.9)',
        'text-background-opacity': 1,
        'text-background-padding': '3px',
        'border-width': 2,
        'border-color': '#555',
        // ✅ NEW: Smooth transitions
        'transition-property': 'background-color, border-color, border-width, width, height',
        'transition-duration': '0.2s',
        'transition-timing-function': 'ease-in-out',
      },
    },
    // Entity nodes - Blue circles
    {
      selector: 'node[type="entity"]',
      style: {
        shape: 'ellipse',
        'background-color': GRAPH_COLORS.entity,
        'border-color': '#2563eb',
      },
    },
    // Relation nodes - Red diamonds (DYNAMIC sizing)
    {
      selector: 'node[type="relation"]',
      style: {
        shape: 'diamond',
        'background-color': GRAPH_COLORS.relation,
        'border-color': '#dc2626',
        // ✅ PHASE 3: Relations slightly larger than entities (dynamic)
        width: (ele: any) => {
          const weight = ele.data('weight') || 0.5;
          const minSize = 22;
          const maxSize = 55;
          const normalized = Math.min(weight, 1.0);
          return minSize + (normalized * (maxSize - minSize));
        },
        height: (ele: any) => {
          const weight = ele.data('weight') || 0.5;
          const minSize = 22;
          const maxSize = 55;
          const normalized = Math.min(weight, 1.0);
          return minSize + (normalized * (maxSize - minSize));
        },
      },
    },
    // Chunk nodes - Green rectangles
    {
      selector: 'node[type="chunk"]',
      style: {
        shape: 'roundrectangle',
        'background-color': GRAPH_COLORS.chunk,
        'border-color': '#059669',
      },
    },
    // Document nodes - Purple rectangles
    {
      selector: 'node[type="document"]',
      style: {
        shape: 'roundrectangle',
        'background-color': GRAPH_COLORS.document,
        'border-color': '#7c3aed',
      },
    },
    // ✅ IMPROVED: Selected node with smooth animation
    {
      selector: 'node:selected',
      style: {
        'border-width': 4,
        'border-color': '#f59e0b',
        'background-color': GRAPH_COLORS.selected,
        'z-index': 999,
      },
    },
    // Hovered node
    {
      selector: 'node.hover',
      style: {
        'border-width': 3,
        'border-color': GRAPH_COLORS.hover,
      },
    },
    // Highlighted node (from search)
    {
      selector: 'node.highlighted',
      style: {
        'border-width': 3,
        'border-color': '#f59e0b',
        'background-color': '#fbbf24',
      },
    },
    // Pulse animation for search navigation
    {
      selector: 'node.pulse',
      style: {
        'border-width': 4,
        'border-color': '#FF6B35',
      },
    },
    // ✅ PHASE 3: Edges with DYNAMIC width based on weight
    {
      selector: 'edge',
      style: {
        // ✅ PHASE 3: Dynamic edge width based on weight (1px - 5px range)
        width: (ele: any) => {
          const weight = ele.data('weight') || 1.0;
          const minWidth = 1;
          const maxWidth = 5;
          // Normalize weight (assuming max ~100)
          const normalized = Math.min(weight / 100, 1.0);
          return minWidth + (normalized * (maxWidth - minWidth));
        },
        'line-color': '#64748b',
        'target-arrow-color': '#64748b',
        'target-arrow-shape': 'triangle' as any,
        'arrow-scale': 1.2,
        'curve-style': 'bezier' as any,
        'line-style': 'solid',
        opacity: 0.7,
        label: '',
        'font-size': '8px',
        'text-background-color': '#fff',
        'text-background-opacity': 1,
        'text-background-padding': '2px',
        // ✅ NEW: Smooth transitions
        'transition-property': 'line-color, width, opacity',
        'transition-duration': '0.2s',
      },
    },
    // Edge on hover - show label
    {
      selector: 'edge:hover',
      style: {
        label: 'data(label)',
        width: 3,
        'line-color': '#475569',
        'target-arrow-color': '#475569',
        opacity: 1,
      },
    },
    // Selected edge
    {
      selector: 'edge:selected',
      style: {
        label: 'data(label)',
        width: 3,
        'line-color': GRAPH_COLORS.selected,
        'target-arrow-color': GRAPH_COLORS.selected,
        opacity: 1,
      },
    },
    // Highlighted edges
    {
      selector: 'edge.highlighted',
      style: {
        width: 3,
        'line-color': '#f59e0b',
        'target-arrow-color': '#f59e0b',
        opacity: 1,
      },
    },
  ];

  // ✅ FIX: Prevent graph reset on node selection
  const handleNodeTap = useCallback((evt: EventObject) => {
    const node = evt.target;
    if (node && node.isNode && node.isNode()) {
      // ✅ FIXED: Don't trigger re-render that causes layout reset
      evt.preventDefault();
      evt.stopPropagation();

      const nodeData: CytoscapeNode = {
        data: node.data(),
        position: node.position(),
      };
      onNodeSelect?.(nodeData);
    }
  }, [onNodeSelect]);

  // Handle background tap (deselect)
  const handleBackgroundTap = useCallback((evt: EventObject) => {
    // Only deselect if clicking on background, not on node
    if (evt.target === cyRef.current) {
      onNodeSelect?.(null);
    }
  }, [onNodeSelect]);

  // Handle node hover
  const handleNodeMouseOver = useCallback((evt: EventObject) => {
    const node = evt.target;
    if (node && node.isNode && node.isNode()) {
      node.addClass('hover');
      const nodeData: CytoscapeNode = {
        data: node.data(),
        position: node.position(),
      };
      onNodeHover?.(nodeData);
    }
  }, [onNodeHover]);

  const handleNodeMouseOut = useCallback((evt: EventObject) => {
    const node = evt.target;
    if (node && node.isNode && node.isNode()) {
      node.removeClass('hover');
      onNodeHover?.(null);
    }
  }, [onNodeHover]);

  // ✅ PHASE 3: Enhanced layout function with all algorithms
  const runLayout = useCallback((cy: Core, layoutName: string) => {
    console.log('[GraphCanvas] Running layout:', layoutName, 'with', cy.nodes().length, 'nodes');

    const layoutOptions = {
      name: layoutName,
      animate: true,
      animationDuration: 1000,
      animationEasing: 'ease-in-out-cubic',
      ...(layoutName === 'cose-bilkent' && {
        idealEdgeLength: 80,
        nodeRepulsion: 200000,
        edgeElasticity: 0.45,
        nestingFactor: 0.1,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
      }),
      ...(layoutName === 'dagre' && {
        rankDir: 'TB',
        nodeSep: 80,        // Increased from 50
        edgeSep: 20,        // Increased from 10
        rankSep: 150,       // Increased from 100
        ranker: 'longest-path',
        align: undefined,
      }),
      ...(layoutName === 'fcose' && {
        idealEdgeLength: 100,      // Increased from 80
        nodeRepulsion: 8000,       // Increased from 4500
        edgeElasticity: 0.45,
        numIter: 3000,             // Increased from 2500
        tile: true,
        quality: 'proof',
        gravity: 0.25,
        gravityRange: 3.8,
      }),
      // ✅ PHASE 3: Cola force-directed layout
      ...(layoutName === 'cola' && {
        nodeSpacing: 80,            // Increased from 50
        edgeLength: 120,            // Increased from 100
        animate: true,
        randomize: false,
        maxSimulationTime: 4000,    // Increased from 3000
        convergenceThreshold: 0.01,
        flow: { axis: 'y', minSeparation: 60 },
      }),
      // ✅ PHASE 3: Concentric radial layout (by weight)
      ...(layoutName === 'concentric' && {
        concentric: (node: any) => {
          const weight = node.data('weight') || 0.5;
          return weight * 100; // Scale weight for better distribution
        },
        levelWidth: () => 3,         // Increased from 2
        minNodeSpacing: 60,          // Increased from 40
        spacingFactor: 1.5,
        equidistant: false,
        startAngle: 0,
        sweep: undefined,
      }),
      // ✅ PHASE 3: Circle layout
      ...(layoutName === 'circle' && {
        radius: undefined,
        spacingFactor: 2.0,          // Increased from 1.5
        startAngle: Math.PI / 4,     // Start at 45 degrees
        sweep: undefined,
      }),
      // ✅ PHASE 3: Breadthfirst tree layout
      ...(layoutName === 'breadthfirst' && {
        directed: true,
        spacingFactor: 2.0,          // Increased from 1.5
        roots: undefined,
        grid: false,
        avoidOverlap: true,
        nodeDimensionsIncludeLabels: true,
      }),
      // ✅ PHASE 3: Grid layout
      ...(layoutName === 'grid' && {
        rows: undefined,
        cols: undefined,
        position: undefined,
        avoidOverlap: true,
        avoidOverlapPadding: 20,
        nodeDimensionsIncludeLabels: true,
        spacingFactor: 1.5,
      }),
    };

    const layoutInstance = cy.layout(layoutOptions as any);

    // ✅ NEW: Auto-fit graph after layout completes
    layoutInstance.on('layoutstop', () => {
      console.log('[GraphCanvas] Layout complete, fitting graph to view');
      cy.fit(undefined, 50); // Fit with 50px padding
    });

    layoutInstance.run();
  }, []);

  // ✅ FIX: Initialize Cytoscape only once
  const handleCyReady = useCallback((cy: Core) => {
    // Skip if already initialized
    if (isInitialized.current) return;

    console.log('[GraphCanvas] Initializing Cytoscape instance');
    cyRef.current = cy;
    isInitialized.current = true;

    // Enable user interactions
    cy.userZoomingEnabled(true);
    cy.userPanningEnabled(true);
    cy.boxSelectionEnabled(true);
    cy.minZoom(0.1);
    cy.maxZoom(3);

    // Set up event listeners
    cy.on('tap', 'node', handleNodeTap);
    cy.on('tap', handleBackgroundTap);
    cy.on('mouseover', 'node', handleNodeMouseOver);
    cy.on('mouseout', 'node', handleNodeMouseOut);

    // Mobile touch support
    cy.on('taphold', 'node', handleNodeTap);

    // ✅ IMPROVED: Double-tap to zoom in on node
    let lastTap = 0;
    cy.on('tap', 'node', (evt: EventObject) => {
      const now = Date.now();
      if (now - lastTap < 300) {
        const node = evt.target;
        if (node && node.isNode && node.isNode()) {
          cy.animate({
            center: { eles: node },
            zoom: Math.min(cy.zoom() * 1.5, 3),
            duration: 300,
            easing: 'ease-in-out-cubic',
          });
        }
      }
      lastTap = now;
    });

    // Call parent's onReady
    onReady?.(cy);

    console.log('[GraphCanvas] Initialization complete');
  }, [handleNodeTap, handleBackgroundTap, handleNodeMouseOver, handleNodeMouseOut, onReady]);

  // ✅ CRITICAL FIX: Run layout when elements are loaded OR layout changes
  useEffect(() => {
    if (!cyRef.current || !isInitialized.current) {
      console.log('[GraphCanvas] Cytoscape not ready yet');
      return;
    }

    if (nodes.length === 0 && edges.length === 0) {
      console.log('[GraphCanvas] No nodes/edges yet, skipping layout');
      return;
    }

    // Check if layout changed
    if (previousLayout.current !== layout) {
      console.log('[GraphCanvas] Layout changed from', previousLayout.current, 'to', layout);
      previousLayout.current = layout;
    }

    // Always run layout when nodes/edges change or layout type changes
    console.log('[GraphCanvas] Running layout with', nodes.length, 'nodes and', edges.length, 'edges');
    runLayout(cyRef.current, layout);
  }, [layout, nodes.length, edges.length, runLayout]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (cyRef.current) {
        cyRef.current.removeAllListeners();
        cyRef.current.destroy();
      }
      isInitialized.current = false;
    };
  }, []);

  // Prepare elements for Cytoscape
  const elements = [...nodes, ...edges];

  // Debug logging
  useEffect(() => {
    console.log('[GraphCanvas] Rendering:', {
      nodes: nodes.length,
      edges: edges.length,
      elements: elements.length,
    });
    if (edges.length > 0 && edges.length < 5) {
      console.log('[GraphCanvas] Sample edges:', edges);
    }
  }, [nodes, edges, elements]);

  return (
    <div className={`w-full h-full ${className} relative`}>
      {/* Professional background with grid pattern */}
      <div className="absolute inset-0 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(156, 163, 175, 0.1) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(156, 163, 175, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '30px 30px'
          }}
        />
      </div>

      {/* Graph canvas */}
      <CytoscapeComponent
        elements={elements}
        stylesheet={stylesheet}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          zIndex: 1,
        }}
        cy={(cy: Core) => handleCyReady(cy)}
        wheelSensitivity={0.2}
      />
    </div>
  );
});

GraphCanvas.displayName = 'GraphCanvas';

export default GraphCanvas;
