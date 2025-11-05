import React, { useEffect, useRef, useCallback, memo } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape, { Core, EventObject } from 'cytoscape';
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

  // Cytoscape stylesheet
  const stylesheet = [
    // Node styles
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'text-valign': 'center' as any,
        'text-halign': 'center' as any,
        'background-color': (ele: any) => {
          const type = ele.data('type');
          return GRAPH_COLORS[type as keyof typeof GRAPH_COLORS] || GRAPH_COLORS.entity;
        },
        width: (ele: any) => 30 + (ele.data('weight') || 0) * 20,
        height: (ele: any) => 30 + (ele.data('weight') || 0) * 20,
        'text-wrap': 'wrap' as any,
        'text-max-width': '100px',
        'font-size': '12px',
        'border-width': 2,
        'border-color': '#666',
      },
    },
    // Entity nodes
    {
      selector: 'node[type="entity"]',
      style: {
        shape: 'ellipse',
        'background-color': GRAPH_COLORS.entity,
      },
    },
    // Relation nodes (bipartite edges in the graph)
    {
      selector: 'node[type="relation"]',
      style: {
        shape: 'diamond',
        'background-color': GRAPH_COLORS.relation,
      },
    },
    // Chunk nodes
    {
      selector: 'node[type="chunk"]',
      style: {
        shape: 'rectangle',
        'background-color': GRAPH_COLORS.chunk,
      },
    },
    // Document nodes
    {
      selector: 'node[type="document"]',
      style: {
        shape: 'roundrectangle',
        'background-color': GRAPH_COLORS.document,
      },
    },
    // Selected node
    {
      selector: 'node:selected',
      style: {
        'background-color': GRAPH_COLORS.selected,
        'border-width': 3,
        'border-color': '#000',
      },
    },
    // Hovered node
    {
      selector: 'node.hover',
      style: {
        'background-color': GRAPH_COLORS.hover,
        'border-width': 3,
        'border-color': '#333',
      },
    },
    // Highlighted node (from search)
    {
      selector: 'node.highlighted',
      style: {
        'border-width': 4,
        'border-color': '#FFD700',
        'background-color': '#FFA500',
      },
    },
    // Edge styles
    {
      selector: 'edge',
      style: {
        width: (ele: any) => 1 + (ele.data('weight') || 0) * 3,
        'line-color': '#ccc',
        'target-arrow-color': '#ccc',
        'target-arrow-shape': 'triangle' as any,
        'curve-style': 'bezier' as any,
        label: 'data(label)',
        'font-size': '10px',
        'text-rotation': 'autorotate' as any,
      },
    },
    // Selected edge
    {
      selector: 'edge:selected',
      style: {
        'line-color': GRAPH_COLORS.selected,
        'target-arrow-color': GRAPH_COLORS.selected,
        width: 3,
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

    // Set up event listeners
    cy.on('tap', 'node', handleNodeTap);
    cy.on('tap', handleBackgroundTap);
    cy.on('mouseover', 'node', handleNodeMouseOver);
    cy.on('mouseout', 'node', handleNodeMouseOut);

    // Enable user interactions
    cy.userZoomingEnabled(true);
    cy.userPanningEnabled(true);
    cy.boxSelectionEnabled(true);

    // Call parent's onReady
    onReady?.(cy);

    // Run initial layout
    const layoutOptions = {
      name: layout,
      animate: true,
      animationDuration: 1000,
      // Layout-specific options
      ...(layout === 'cose-bilkent' && {
        idealEdgeLength: 100,
        nodeRepulsion: 400000,
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
        idealEdgeLength: 100,
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

  return (
    <div className={`w-full h-full ${className}`}>
      <CytoscapeComponent
        elements={elements}
        stylesheet={stylesheet}
        style={{ width: '100%', height: '100%' }}
        cy={(cy) => handleCyReady(cy)}
        wheelSensitivity={0.2}
      />
    </div>
  );
});

GraphCanvas.displayName = 'GraphCanvas';

export default GraphCanvas;