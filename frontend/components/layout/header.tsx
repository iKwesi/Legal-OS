'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function Header() {
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <header className="flex items-center justify-between px-6 sm:px-10 py-4 border-b border-border bg-white">
      {/* Left: Logo and App Name */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center size-8 bg-primary rounded-full text-white">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20Z" />
          </svg>
        </div>
        <h1 className="text-xl font-bold text-gray-900">LegalAI</h1>
      </div>

      {/* Center: Navigation Links */}
      <nav className="hidden md:flex items-center gap-8">
        <Link
          href="/"
          className="text-sm font-medium hover:text-primary transition-colors"
        >
          Dashboard
        </Link>
        <Link
          href="/"
          className="text-sm font-medium hover:text-primary transition-colors"
        >
          Projects
        </Link>
        <Link
          href="/"
          className="text-sm font-medium hover:text-primary transition-colors"
        >
          Documents
        </Link>
        <Link
          href="/"
          className="text-sm font-medium hover:text-primary transition-colors"
        >
          Help
        </Link>
      </nav>

      {/* Right: User Avatar */}
      <div className="flex items-center gap-4">
        <div 
          className="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10"
          style={{
            backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuAbmMbv8TdjmQCGcH1BIPscBrGh3riU1cHJ873xnrYdixC8BMS0LZRpVQLCygAGIIvN3mhOrBP06UQCz-radRPYzpIasaPdGnA-pvwLB_xC4O7eD81lL27ePNkL53xAR5jV_oy8Eywv8ZvYZHH4be-Qmh8PC3b1L65EvVTr08AuOGz_Dh_rcIGY8U2A7ysXHwbLkM0eXgojYHkKDOlkCv2O2d-CZhzYOzh7Axv6GDgaaX_wKbGMx6fEvCO68Q5oYlTIvn6RyHk8Lm8")'
          }}
        />
      </div>
    </header>
  );
}
