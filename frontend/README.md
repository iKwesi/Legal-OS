# Legal-OS Frontend

Modern Next.js frontend for Legal-OS with interactive chat, document upload, and analysis results visualization.

## 🏗️ Architecture

The frontend is built with Next.js 15 using the App Router and organized as follows:

- **`app/`** - Next.js App Router pages and layouts
  - `app/page.tsx` - Home/upload page
  - `app/chat/` - Interactive chat interface
  - `app/checklist/` - Checklist view
  - `app/results/` - Analysis results display
- **`components/`** - React components
  - `components/chat/` - Chat UI components
  - `components/upload/` - Document upload components
  - `components/results/` - Results display components
  - `components/ui/` - Shadcn/ui base components
- **`lib/`** - Utilities and API client
- **`types/`** - TypeScript type definitions

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Set up environment (if needed)
# The API URL defaults to http://localhost:8000

# Run development server
npm run dev
```

Access the app at: http://localhost:3000

### Docker

```bash
# From project root
docker-compose up -d frontend
```

## 🎨 UI Components

### Pages

#### Home/Upload Page (`app/page.tsx`)
- Document upload interface
- Drag-and-drop support
- Upload progress tracking
- Recent documents list

#### Chat Interface (`app/chat/page.tsx`)
- Interactive chat with uploaded documents
- Message history
- Source citations
- Context-aware responses

#### Checklist View (`app/checklist/page.tsx`)
- Due diligence checklist display
- Priority filtering
- Status tracking
- Category organization

#### Results Page (`app/results/page.tsx`)
- Comprehensive analysis results
- Clause extraction display
- Risk scoring visualization
- Summary presentation

### Component Library

#### Upload Components (`components/upload/`)

**FileUpload.tsx**
- Drag-and-drop file upload
- File validation (PDF only)
- Upload progress indicator
- Error handling

**DocumentList.tsx**
- List of uploaded documents
- Document metadata display
- Quick actions (view, analyze, delete)

#### Chat Components (`components/chat/`)

**ChatInterface.tsx**
- Main chat container
- Message threading
- Auto-scroll to latest message

**MessageBubble.tsx**
- User and assistant message display
- Timestamp formatting
- Source citation links

**ChatInput.tsx**
- Message input field
- Send button
- Character count (optional)
- Loading state

**SourceCard.tsx**
- Source document citation
- Relevance score display
- Quick preview

#### Results Components (`components/results/`)

**AnalysisOverview.tsx**
- High-level analysis summary
- Key metrics display
- Navigation to detailed sections

**ClauseList.tsx**
- Extracted clauses display
- Filtering by clause type
- Confidence scores

**RiskMatrix.tsx**
- Risk visualization
- Severity indicators
- Category breakdown

**SummaryCard.tsx**
- Document summary display
- Key points extraction
- Executive summary

#### UI Components (`components/ui/`)

Shadcn/ui components (Radix primitives + Tailwind):
- `button.tsx` - Button variants
- `card.tsx` - Card container
- `input.tsx` - Form inputs
- `badge.tsx` - Status badges
- `alert.tsx` - Alert messages
- `progress.tsx` - Progress bars
- `skeleton.tsx` - Loading skeletons
- `toast.tsx` - Toast notifications
- And more...

## 🔧 Configuration

### Environment Variables

Create `.env.local` for local development:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Analytics, monitoring, etc.
# NEXT_PUBLIC_GA_ID=your-ga-id
```

### API Client (`lib/api.ts`)

The API client provides typed methods for all backend endpoints:

```typescript
import { api } from '@/lib/api';

// Upload document
const result = await api.uploadDocument(file);

// Query documents
const response = await api.query({
  query: "What are the key terms?",
  document_id: "uuid"
});

// Chat
const chatResponse = await api.chat({
  message: "Tell me about risks",
  document_id: "uuid"
});

// Get analysis
const analysis = await api.orchestrate({
  document_id: "uuid",
  query: "Perform full analysis"
});
```

## 🎯 Features

### Document Upload
- **Drag & Drop**: Intuitive file upload
- **Validation**: PDF format validation
- **Progress**: Real-time upload progress
- **Feedback**: Success/error notifications

### Interactive Chat
- **Context-Aware**: Understands document context
- **Source Citations**: Links to relevant document sections
- **Multi-Turn**: Maintains conversation history
- **Streaming**: Real-time response streaming (if enabled)

### Analysis Results
- **Comprehensive**: All agent results in one view
- **Visual**: Charts and graphs for risk scores
- **Filterable**: Filter by category, priority, etc.
- **Exportable**: Download results as PDF/JSON

### Responsive Design
- **Mobile-First**: Works on all screen sizes
- **Touch-Friendly**: Optimized for touch interactions
- **Accessible**: WCAG 2.1 AA compliant

## 🧪 Testing

### Run Tests
```bash
npm run test
```

### Run Tests in Watch Mode
```bash
npm run test:watch
```

### Run E2E Tests (if configured)
```bash
npm run test:e2e
```

## 🎨 Styling

### Tailwind CSS

The project uses Tailwind CSS for styling with a custom configuration:

```javascript
// tailwind.config.ts
{
  theme: {
    extend: {
      colors: {
        // Custom color palette
      },
      // Custom spacing, fonts, etc.
    }
  }
}
```

### Design System

- **Colors**: Consistent color palette from Shadcn/ui
- **Typography**: System font stack with fallbacks
- **Spacing**: 4px base unit (Tailwind default)
- **Breakpoints**: Mobile-first responsive breakpoints

## 📱 Responsive Breakpoints

```css
sm: 640px   /* Small devices */
md: 768px   /* Medium devices */
lg: 1024px  /* Large devices */
xl: 1280px  /* Extra large devices */
2xl: 1536px /* 2X large devices */
```

## 🔍 Code Quality

### Linting
```bash
npm run lint
```

### Type Checking
```bash
npm run type-check
# or
npx tsc --noEmit
```

### Formatting
```bash
npm run format
# or
npx prettier --write .
```

## 📝 Development Guidelines

### Component Structure

```typescript
// components/example/ExampleComponent.tsx
import { FC } from 'react';

interface ExampleComponentProps {
  title: string;
  onAction?: () => void;
}

export const ExampleComponent: FC<ExampleComponentProps> = ({
  title,
  onAction
}) => {
  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">{title}</h2>
      {onAction && (
        <button onClick={onAction}>Action</button>
      )}
    </div>
  );
};
```

### Best Practices

1. **TypeScript**: Use strict type checking
2. **Components**: Keep components small and focused
3. **Hooks**: Extract complex logic into custom hooks
4. **Styling**: Use Tailwind utility classes
5. **Accessibility**: Include ARIA labels and keyboard navigation
6. **Performance**: Use React.memo for expensive components
7. **Error Handling**: Implement error boundaries

### File Naming

- **Components**: PascalCase (e.g., `ChatInterface.tsx`)
- **Utilities**: camelCase (e.g., `formatDate.ts`)
- **Types**: PascalCase (e.g., `ChatMessage.ts`)
- **Pages**: lowercase with hyphens (e.g., `chat/page.tsx`)

## 🚀 Build & Deploy

### Production Build
```bash
npm run build
```

### Start Production Server
```bash
npm run start
```

### Analyze Bundle
```bash
npm run analyze
```

## 🐛 Troubleshooting

### Common Issues

**Module not found:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json .next
npm install
```

**Type errors:**
```bash
# Regenerate types
npm run type-check
```

**Styling issues:**
```bash
# Rebuild Tailwind
npm run dev
# or
npx tailwindcss -i ./app/globals.css -o ./dist/output.css
```

**API connection errors:**
- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Inspect network tab for CORS errors

**Hot reload not working:**
- Restart dev server
- Check file watcher limits (Linux)
- Clear `.next` cache

## 📦 Dependencies

### Core
- **Next.js 15** - React framework
- **React 18** - UI library
- **TypeScript** - Type safety

### UI
- **Tailwind CSS** - Utility-first CSS
- **Shadcn/ui** - Component library
- **Radix UI** - Accessible primitives
- **Lucide React** - Icon library

### State Management
- **Zustand** - Lightweight state management
- **React Context** - Built-in state sharing

### Utilities
- **clsx** - Conditional classnames
- **date-fns** - Date formatting
- **zod** - Schema validation

## 📖 Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Shadcn/ui Documentation](https://ui.shadcn.com)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)

## 🤝 Contributing

1. Follow the component structure guidelines
2. Write TypeScript with strict types
3. Add tests for new features
4. Use Tailwind for styling
5. Ensure accessibility compliance
6. Update documentation as needed

## 📄 License

[Your License Here]
