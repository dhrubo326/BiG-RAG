// Type declarations for Cytoscape modules

declare module 'react-cytoscapejs' {
  import { Component } from 'react';
  import { Core, ElementDefinition, Stylesheet } from 'cytoscape';

  interface CytoscapeComponentProps {
    elements: ElementDefinition[];
    stylesheet?: Stylesheet[];
    style?: React.CSSProperties;
    cy?: (cy: Core) => void;
    wheelSensitivity?: number;
  }

  export default class CytoscapeComponent extends Component<CytoscapeComponentProps> {}
}

declare module 'cytoscape-cose-bilkent' {
  const coseBilkent: any;
  export default coseBilkent;
}

declare module 'cytoscape-dagre' {
  const dagre: any;
  export default dagre;
}

declare module 'cytoscape-fcose' {
  const fcose: any;
  export default fcose;
}