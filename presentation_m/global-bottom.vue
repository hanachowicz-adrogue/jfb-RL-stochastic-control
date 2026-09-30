<script setup lang="ts">
import { useNav } from '@slidev/client'
const { currentPage } = useNav()
</script>

<template>
  <span class="deck-footer">{{ currentPage <= 9 ? `${currentPage} / 9` : `Appendix ${currentPage - 9}` }}</span>
</template>
