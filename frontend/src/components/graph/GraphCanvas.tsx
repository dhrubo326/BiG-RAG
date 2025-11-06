import React, { useEffect, useRef, useCallback, memo, useState } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape, { type Core, type EventObject } from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import dagre from 'cytoscape-dagre';
import fcose from 'cytoscape-fcose';
import { GRAPH_COLORS } from '../../utils/constants';
import type { CytoscapeNode, CytoscapeEdge } from '../../types';

// Register layout algorithms
cytoscape.use(coseBilkent);
cytoscape.use(dagre);
cytoscape.use(fcose);

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

  // ✅ FIXED: Simple, clean stylesheet with SMALL nodes and VISIBLE edges
  const stylesheet = [
    // Default node styles - MUCH SMALLER
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'text-valign': 'bottom' as any,
        'text-halign': 'center' as any,
        'text-margin-y': 3,
        'background-color': (ele: any) => {
          const type = ele.data('type');
          return GRAPH_COLORS[type as keyof typeof GRAPH_COLORS] || GRAPH_COLORS.entity;
        },
        // ✅ FIXED: Much smaller nodes - 20-25px only!
        width: 20,
        height: 20,
        // ✅ FIXED: Smaller text to match
        'text-wrap': 'wrap' as any,
        'text-max-width': '60px',
        'font-size': '9px',
        'font-weight': '600',
        'color': '#111',
        'text-outline-color': '#fff',
        'text-outline-width': 1,
        'text-background-color': 'rgba(255, 255, 255, 0.8)',
        'text-background-opacity': 1,
        'text-background-padding': '2px',
        // ✅ FIXED: Thin borders
        'border-width': 1.5,
        'border-color': '#555',
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
    // Relation nodes - Red diamonds
    {
      selector: 'node[type="relation"]',
      style: {
        shape: 'diamond',
        'background-color': GRAPH_COLORS.relation,
        'border-color': '#dc2626',
        width: 22, // Slightly larger for visibility
        height: 22,
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
    // Selected node
    {
      selector: 'node:selected',
      style: {
        'border-width': 3,
        'border-color': '#000',
        'background-color': GRAPH_COLORS.selected,
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
    // ✅ FIXED: THICK, VISIBLE edges!
    {
      selector: 'edge',
      style: {
        width: 2, // Thick enough to see clearly
        'line-color': '#64748b', // Darker slate gray - very visible!
        'target-arrow-color': '#64748b',
        'target-arrow-shape': 'triangle' as any,
        'arrow-scale': 1.2,
        'curve-style': 'bezier' as any,
        'line-style': 'solid',
        opacity: 0.7,
        label: '', // Hide labels by default
        'font-size': '8px',
        'text-background-color': '#fff',
        'text-background-opacity': 1,
        'text-background-padding': '2px',
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

  // Handle node selection
  const handleNodeTap = useCallback((evt: EventObject) => {
    const node = evt.target;
    if (node && node.isNode && node.isNode()) {
      const nodeData: CytoscapeNode = {
        data: node.data(),
        position: node.position(),
      };
      onNodeSelect?.(nodeData);
    }
  }, [onNodeSelect]);

  // Handle background tap (deselect)
  const handleBackgroundTap = useCallback(() => {
    onNodeSelect?.(null);
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

  // Initialize Cytoscape
  const handleCyReady = useCallback((cy: Core) => {
    cyRef.current = cy;

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

    // Double-tap to zoom
    let lastTap = 0;
    cy.on('tap', 'node', (evt: EventObject) => {
      const now = Date.now();
      if (now - lastTap < 300) {
        const node = evt.target;
        if (node && node.isNode && node.isNode()) {
          cy.animate({
            center: { eles: node },
            zoom: cy.zoom() * 1.5,
            duration: 300,
          });
        }
      }
      lastTap = now;
    });

    // Call parent's onReady
    onReady?.(cy);

    // Run initial layout
    const layoutOptions = {
      name: layout,
      animate: true,
      animationDuration: 1000,
      ...(layout === 'cose-bilkent' && {
        idealEdgeLength: 80,
        nodeRepulsion: 200000,
        edgeElasticity: 0.45,
        nestingFactor: 0.1,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
      }),
      ...(layout === 'dagre' && {
        rankDir: 'TB',
        nodeSep: 50,
        edgeSep: 10,
        rankSep: 100,
      }),
      ...(layout === 'fcose' && {
        idealEdgeLength: 80,
        nodeRepulsion: 4500,
        edgeElasticity: 0.45,
        numIter: 2500,
        tile: true,
      }),
    };

    cy.layout(layoutOptions as any).run();
  }, [handleNodeTap, handleBackgroundTap, handleNodeMouseOver, handleNodeMouseOut, onReady, layout]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (cyRef.current) {
        cyRef.current.removeAllListeners();
      }
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
    if (edges.length > 0) {
      console.log('[GraphCanvas] Sample edge:', edges[0]);
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
