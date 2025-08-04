# Layout Components

This directory contains the main layout components for the myQuant frontend application.

## Components

### MainLayout
The main layout wrapper that provides the overall structure of the application.

**Features:**
- Responsive design with mobile support
- Keyboard navigation (Ctrl/Cmd + B to toggle sidebar)
- Error boundary integration
- Customizable breadcrumb and footer display

**Props:**
- `children`: React.ReactNode - The main content to display
- `showBreadcrumb`: boolean - Whether to show breadcrumb navigation (default: true)
- `showFooter`: boolean - Whether to show footer (default: true)
- `sidebarCollapsed`: boolean - Control sidebar collapse state externally
- `onSidebarCollapse`: (collapsed: boolean) => void - Callback for sidebar state changes

### Header
The top navigation header with system controls and user menu.

**Features:**
- System status indicator with pulse animation
- Theme toggle (light/dark)
- Language selector (placeholder for i18n)
- Notification center
- User profile menu with settings
- Responsive design for mobile

**Props:**
- `collapsed`: boolean - Current sidebar collapse state
- `onCollapse`: (collapsed: boolean) => void - Callback to toggle sidebar
- `isMobile`: boolean - Whether the current view is mobile

### Sidebar
The left navigation sidebar with menu items.

**Features:**
- Responsive navigation menu
- Auto-collapse on mobile after navigation
- Mobile overlay for better UX
- Keyboard navigation support
- Custom scrollbar styling

**Props:**
- `collapsed`: boolean - Whether the sidebar is collapsed
- `onCollapse`: (collapsed: boolean) => void - Callback to toggle collapse
- `isMobile`: boolean - Whether the current view is mobile

### Footer
The bottom footer with links and copyright information.

**Features:**
- Customizable links and content
- Responsive design
- Theme-aware styling

**Props:**
- `showLinks`: boolean - Whether to show footer links (default: true)
- `showCopyright`: boolean - Whether to show copyright info (default: true)
- `customContent`: React.ReactNode - Custom content to replace default footer

### Breadcrumb
Dynamic breadcrumb navigation based on current route.

**Features:**
- Auto-generation from route path
- Custom breadcrumb items support
- Home link toggle
- Custom separator support

**Props:**
- `customItems`: BreadcrumbItem[] - Custom breadcrumb items
- `showHome`: boolean - Whether to show home link (default: true)
- `separator`: string - Custom separator (default: '/')

### ThemeProvider
Global theme provider with dark/light mode support.

**Features:**
- Ant Design theme configuration
- Chart theme synchronization
- CSS custom properties management
- Theme persistence

### SettingsModal
System settings modal for user preferences.

**Features:**
- Theme selection
- Language settings (placeholder)
- Auto-refresh configuration
- Notification preferences
- Compact mode toggle
- Animation controls

**Props:**
- `visible`: boolean - Whether the modal is visible
- `onClose`: () => void - Callback when modal is closed

## Usage

```tsx
import { MainLayout, ThemeProvider } from './components/Layout';

function App() {
  return (
    <ThemeProvider>
      <MainLayout>
        <YourPageContent />
      </MainLayout>
    </ThemeProvider>
  );
}
```

## Keyboard Shortcuts

- `Ctrl/Cmd + B`: Toggle sidebar collapse

## Responsive Breakpoints

- Mobile: < 768px
- Desktop: >= 768px

## Theme Support

The layout components support both light and dark themes through the ThemeProvider. Theme changes are automatically applied to:

- Ant Design components
- Chart components (ECharts)
- CSS custom properties
- Layout-specific styling

## Accessibility

All layout components include accessibility features:

- ARIA labels and roles
- Keyboard navigation support
- Focus management
- Screen reader compatibility
- High contrast support

## Testing

Tests are located in the `__tests__` directory and use Vitest with React Testing Library.

Run tests with:
```bash
npm test Layout
```

## Styling

Custom styles are defined in `Layout.css` and include:

- Responsive design rules
- Theme-specific styling
- Animation definitions
- Accessibility improvements
- Custom scrollbar styling